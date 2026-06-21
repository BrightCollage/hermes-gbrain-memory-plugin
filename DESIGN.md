# Brain-Agent Loop: Reference Design

> How to wire persistent knowledge into any AI agent so every conversation
> makes the brain smarter and every brain lookup makes responses better.

**Source of truth:** [garrytan/gbrain/docs/guides/brain-agent-loop.md](https://github.com/garrytan/gbrain/blob/master/docs/guides/brain-agent-loop.md)
**Reference implementation:** [BrightCollage/hermes-gbrain-memory-plugin](https://github.com/BrightCollage/hermes-gbrain-memory-plugin/tree/brain-agent-loop)

---

## Table of Contents

1. [Concept: The Loop](#1-concept-the-loop)
2. [Interface Contract](#2-interface-contract)
3. [Implementation: DETECT](#3-implement-detect)
4. [Implementation: READ (prefetch)](#4-implement-read)
5. [Implementation: WRITE (sync)](#5-implement-write)
6. [Orchestration: How the Agent Calls You](#6-orchestration)
7. [The Agent Instruction File: SOUL.md Integration](#7-the-agent-instruction-file-soulmd-integration)
8. [Platform Migration Guide](#8-platform-migration-guide)
9. [Verification](#9-verification)

---

## 1. Concept: The Loop

Every message triggers a **three-phase cycle** within a single conversation turn:

```
Signal arrives (message, email, tweet, link)
  │
  ▼
[DETECT]  Extract entity candidates via stop-word mask
  │         (no NER, no spaCy, zero deps — punctuation-strip + set lookup)
  │
  ▼
[READ]    Search the brain BEFORE composing the response
  │         → gbrain search "{candidate}" (cap 3 lookups)
  │         → inject compiled truth as <memory-context> fence
  │
  ▼
[RESPOND] Compose answer WITH brain context
  │         (every answer is better with context)
  │
  ▼
[WRITE]   After turn, in background thread:
  │         → resolve candidates → person/company pages
  │         → add_timeline_entry for each matched entity
  │         (GBrain indexes server-side — no explicit sync call)
  │
  ▼
(next signal — agent is now smarter)
```

> **Note:** The original guide includes an explicit `gbrain sync` step. Our
> implementation skips it — the GBrain MCP server handles indexing internally.
> The plugin only writes data (timeline entries); sync is the server's job.

**Two invariants:**

1. **Every READ improves the response.** If you answered a question about a
   person without checking their brain page first, you gave a worse answer
   than you could have.
2. **Every WRITE improves future reads.** If a conversation mentioned new
   information and you didn't record it, you created a gap that compounds.

---

## 2. Interface Contract

A memory provider that implements the brain-agent-loop MUST expose these
methods. This is the contract between the agent framework and the brain.

```python
# ──────────────────────────────────────────────
# Phase 1 + 2: DETECT + READ (synchronous)
# Called BEFORE the LLM API call, within the
# same turn. Return value is injected into the
# model's system prompt.
# ──────────────────────────────────────────────
def prefetch(self, query: str, *, session_id: str = "") -> str:
    """
    Search the brain for entities in the current message.

    - Extract entity names from `query`
    - Search the brain for each entity (cap at N lookups)
    - Return formatted context block, or "" if nothing found
    - MUST be fast (< 1s typical); the user is waiting
    - MUST NOT raise exceptions (failures degrade silently)
    """

# ──────────────────────────────────────────────
# Phase 3: WRITE (asynchronous / background)
# Called AFTER the turn completes. Must not block
# the user from seeing the response.
# ──────────────────────────────────────────────
def sync_turn(self,
    user_content: str,
    assistant_content: str,
    *,
    session_id: str = "",
    messages=None,
) -> None:
    """
    Record what was discussed in the completed turn.

    - Extract entity names from user_content
    - For each known entity, add a timeline entry
      marking it as mentioned in this conversation
    - Runs on a background thread — network/IO is OK
    - MUST NOT raise exceptions
    """
```

### What the framework does with the result

The `MemoryManager` in the Hermes agent framework wraps the return value:

```python
def build_memory_context_block(raw_context: str) -> str:
    """Wrap prefetched memory in a fenced block with system note."""
    if not raw_context or not raw_context.strip():
        return ""
    clean = sanitize_context(raw_context)
    return (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data — "
        "this is the agent's persistent memory and should inform all responses.]\n\n"
        f"{clean}\n"
        "</memory-context>"
    )
```

This fenced block is appended to the **current turn's user message** before
sending to the LLM API. It's ephemeral — injected at API-call time, never
persisted to the conversation history.

---

## 3. Implementation: DETECT

Entity detection is the gate to everything else. The reference implementation
uses a **mask-based approach** (no spaCy, no NER model, zero dependencies):

### Algorithm (mask-based)

```
_input: raw message text

1. Split on whitespace into tokens.
2. Strip leading/trailing punctuation from each token
   (keep internal hyphens and apostrophes).
3. For each token:
   a. lower-case it and check against a stop-word set
      (2,226 English words: articles, pronouns, prepositions,
      conjunctions, verbs, adjectives, adverbs, etc.)
   b. If NOT a stop word: accumulate into current_phrase
   c. If IS a stop word: flush accumulated phrase as a candidate
4. Flush final phrase at end.
5. Deduplicate (case-insensitive).
6. Return top N (capped at 10).
```

```python
_STOP_WORDS: set[str]  # 2,226 common English function words
                        # loaded from _stop_words.py

def _extract_candidates(text: str) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    current_phrase: list[str] = []

    for token in text.split():
        clean = token.strip(".,!?;:()[]{}«»…`´~@#$%^&*=+<>/\\|¬·–—\"")
        if not clean:
            continue
        if clean.lower() in _STOP_WORDS:
            if current_phrase:
                phrase = " ".join(current_phrase)
                if phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    candidates.append(phrase)
                current_phrase = []
        else:
            current_phrase.append(clean)

    if current_phrase:
        phrase = " ".join(current_phrase)
        if phrase.lower() not in seen:
            candidates.append(phrase)

    return candidates[:10]
```

### Why not NER?

| Approach | Pros | Cons |
|----------|------|------|
| spaCy / NER model | High accuracy | Heavy dependency, slow to load, GPU optional |
| LLM-based extraction | Best quality | 100ms+ latency, costs tokens on every message |
| Mask-based (this) | Zero deps, ~1µs per message | Misses single-word names ("Pedro" vs "Pedro Franceschi"), catches noise |

The mask approach is a **minimum viable detector**. It catches multi-word
entities reliably ("Acme Corp", "SAS Institute", "Blue Cross NC") which is
where the brain is most valuable. Single-word names can be caught by a
better detector later — the important thing is the loop exists at all.

### Higher-quality alternatives for v2

- Run a cheap model (`sonnet-class`) as a background sub-agent
- Use `gbrain search` directly as a detector (search every n-gram?)
- Accept a small NER model dependency

---

## 4. Implementation: READ (prefetch)

The READ phase runs **synchronously before the LLM API call**. The user is
waiting, so speed matters.

### Reference implementation

```python
def prefetch(self, query: str, *, session_id: str = "") -> str:
    if not self._mcp or not query:
        return ""

    # Phase 1: detect candidates
    candidates = self._extract_candidates(query)
    if not candidates:
        return ""

    # Phase 2: search brain, cap at 3 lookups
    context_parts = []
    for name in candidates[:3]:
        ctx = self._search_entity(name)
        if ctx:
            context_parts.append(ctx)

    if not context_parts:
        return ""

    # Phase 3: format as one block
    return (
        "## GBrain — automatically detected context\n\n"
        + "\n---\n\n".join(context_parts)
    )
```

### `_search_entity` behavior

1. Call `gbrain_search({name}, limit=3)` via `self._mcp.call("search", ...)`
2. Iterate over `content` items in the response
3. **JSON path:** Try to parse each item as `list[{slug, title, chunk_text, type}]`:
   - Take top 2 matches
   - Format as `**{title}** (/{slug}, {type})\n{chunk_text}`
   - Truncate each chunk at **600 chars**
4. **Fallback:** If JSON parsing fails, return the raw text (capped at **1200 chars**)
5. Return `""` if nothing found or any error occurs (silent failure)

All exceptions are caught and logged at DEBUG level so a broken brain
never surfaces to the user.

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Synchronous, not deferred** | The guide says "Read BEFORE responding." Caching for next turn loses the current turn's benefit. |
| **Cap at 3 lookups** | Keeps latency ~1s worst-case. 3 lookups × 300ms each ≈ 900ms. DETECT itself is ~1µs. |
| **Search only, no get-page** | Search is faster and returns compiled_truth directly. Deep reads can happen via agent tools. |
| **Stop-word mask, no NER** | Zero dependencies. Can be swapped for better detector later without changing the loop. |
| **No new entity creation** | Only reads existing brain data. Creating pages is the agent's job through `gbrain_put_page`. |
| **Silent failure** | Any exception → return "". The user never sees a broken brain. |
| **Hardcoded tool schemas** | 83 schemas baked into `_schemas.py` from a live-server dump. No dynamic discovery — saves MCP `tools/list` call on every startup. Schemas are the GBrain `search`/`get`/`put` etc. remapped to `gbrain_*` names. |

---

## 5. Implementation: WRITE (sync_turn)

The WRITE phase runs **after the turn completes**, on a background thread.
The user has already seen the response — latency doesn't matter.

### Reference implementation

```python
def sync_turn(self,
    user_content: str,
    assistant_content: str,
    *,
    session_id: str = "",
    messages=None,
) -> None:
    if not self._mcp or not user_content:
        return

    today_str = date.today().isoformat()
    try:
        entities = self._resolve_entities(user_content)
        for e in entities:
            self._write_timeline(e["slug"], today_str)
    except Exception:
        pass  # silent failure


def _resolve_entities(self, text: str) -> list[dict]:
    """
    Extract entity names and resolve to brain slugs + types.

    Returns list of {name, slug, type} dicts — only resolved
    person/company pages get timeline entries.
    """
    candidates = self._extract_candidates(text)
    if not candidates:
        return []

    resolved: list[dict] = []
    seen_slugs: set[str] = set()

    for name in candidates[:5]:                     # cap at 5 searches
        result = self._mcp.call("search",
                                {"query": name, "limit": 3})
        content_texts = [
            c.get("text", "")
            for c in result.get("content", [])
            if c.get("text")
        ]
        for raw in content_texts:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    for item in data:
                        # Only person/company pages get timeline entries
                        page_type = item.get("type", "")
                        if page_type not in ("person", "company"):
                            continue
                        slug = item.get("slug", "")
                        title = item.get("title", "")
                        if not slug or slug in seen_slugs:
                            continue
                        # Verify candidate name matches page title/slug
                        name_lower = name.lower()
                        title_lower = title.lower()
                        if not (name_lower in title_lower
                                or title_lower.startswith(name_lower)
                                or name_lower.replace(" ", "-")
                                   in slug.lower()):
                            continue
                        resolved.append({
                            "name": name,
                            "slug": slug,
                            "type": page_type,
                        })
                        seen_slugs.add(slug)
            except (json.JSONDecodeError, TypeError):
                # Fallback: extract slugs from wiki-link markdown
                for match in re.finditer(
                    r'\[([^\]]+)\]\(/([^)]+)\)', raw
                ):
                    slug = match.group(2)
                    if slug not in seen_slugs:
                        resolved.append({
                            "name": name,
                            "slug": slug,
                            "type": "unknown",
                        })
                        seen_slugs.add(slug)

    return resolved[:5]  # cap at 5 resolved entities


def _write_timeline(self, slug: str, today_str: str) -> None:
    """Add a 'Mentioned in conversation' timeline entry."""
    try:
        self._mcp.call("add_timeline_entry", {
            "slug": slug,
            "date": today_str,
            "summary": f"Mentioned in conversation on {today_str}",
            "source": "hermes-brain-loop",
        })
        logger.info("brain-loop: added timeline to /%s", slug)
    except Exception as e:
        logger.debug("brain-loop: timeline write failed for /%s: %s",
                     slug, e)
```

### `queue_prefetch` — explicit no-op

```python
def queue_prefetch(self, query: str, *,
                   session_id: str = "") -> None:
    # No-op: the primary read happens synchronously in prefetch().
    # This hook is left available for future cache-warming use.
    pass
```

The Hermes framework calls `queue_prefetch_all()` after each turn on a
background thread. Our implementation deliberately leaves it as a no-op
because the READ phase already runs synchronously on every turn — there's
nothing to warm for "next time." This saves a network round-trip and avoids
confusing the brain with stale prefetches.

### Phased write strategy

| Phase | What to write | When |
|-------|--------------|------|
| **Phase 1 (implemented)** | Timeline entries (`add_timeline_entry`) with "Mentioned in conversation" | Every turn |
| **Phase 2 (future)** | Update compiled truth on pages that got new factual info | When agent detects new info in response |
| **Phase 3 (future)** | Create new entity pages for previously unknown entities | When agent decides entity is notable |

Phase 1 is the minimum viable write — it ensures every entity that was discussed
gets a trail, so the brain compounds purely from conversation.

---

## 6. Orchestration

This is how the agent framework orchestrates the three phases. The pattern is
platform-agnostic — any AI agent can implement it.

### Sequence diagram

```
┌──────────┐    ┌──────────────┐    ┌──────────┐
│  Agent   │    │MemoryManager │    │  Brain   │
│  Loop    │    │ (orchestr.)  │    │ Provider │
└────┬─────┘    └──────┬───────┘    └────┬─────┘
     │                 │                  │
     │── user msg ────►│                  │
     │                 │                  │
     │                 │── prefetch() ───►│  ◄── DETECT + READ
     │                 │                  │     (synchronous)
     │                 │◄── context str ──│
     │                 │                  │
     │◄── context ─────│                  │
     │                 │                  │
     │                 │                  │
     │──── LLM API call with ────────────│──────
     │      <memory-context> injected     │
     │                                    │
     │◄── response ───────────────────────│
     │                 │                  │
     │                 │── sync_turn() ──►│  ◄── WRITE
     │                 │                  │     (background thread)
     │                 │                  │
     │ show response ──┼──────────────────│
```

### Real Hermes agent loop behavior

```python
# Called once per turn, in the conversation loop

# ── Phase 1 + 2: DETECT + READ ──────
# Before any LLM call, fetch brain context
ext_prefetch_cache = ""
if agent._memory_manager:
    try:
        _query = original_user_message  # raw string
        ext_prefetch_cache = (
            agent._memory_manager.prefetch_all(_query) or ""
        )
    except Exception:
        pass

# ── Build API messages ───────────────
# The prefetch context is injected into the current turn's
# user message at API-call time (not persisted to messages[])
if _ext_prefetch_cache:
    _fenced = build_memory_context_block(_ext_prefetch_cache)
    if _fenced:
        api_msg["content"] += "\n\n" + _fenced

# ── LLM API call ─────────────────────
response = llm.chat(api_messages)

# ── Phase 3: WRITE (after turn) ──────
_on_turn_complete(user_msg, response)
```

### `_on_turn_complete` — the real interruption guard

```python
def _on_turn_complete(self,
    final_response, original_user_message,
    messages=None, interrupted=False
) -> None:
    # NEVER sync interrupted turns — response is incomplete
    if interrupted:
        return
    if not (self._memory_manager and final_response
            and original_user_message):
        return

    # Flatten multimodal content to plain text
    user_text = _summarize_user_message_for_log(
        original_user_message, sep="\n")
    response_text = _summarize_user_message_for_log(
        final_response, sep="\n")
    if not (user_text and response_text):
        return

    try:
        sync_kwargs = {"session_id": self.session_id or ""}
        if messages is not None:
            sync_kwargs["messages"] = messages
        self._memory_manager.sync_all(
            user_text, response_text, **sync_kwargs)
        self._memory_manager.queue_prefetch_all(
            user_text, session_id=self.session_id or "")
    except Exception:
        pass  # external memory is strictly best-effort
```

Key differences from the simplified version:
1. **`_strip_skill_scaffolding`**: Before passing to providers, the
   `MemoryManager` strips skill/bundle scaffolding from the user message,
   keeping only the user's actual instruction. Bare skill invocations (no
   instruction) are skipped entirely.
2. **`try/except Exception`** wraps the entire sync — a misconfigured or
   offline backend must never block the user from seeing their response.
3. **Interrupted turns skip sync entirely** — writing incomplete responses
   would pollute the brain with junk the user never saw.

### Thread safety notes

- `prefetch_all()` runs on the **main conversation thread**. It must be fast
  and non-blocking. The user is waiting for the first token.
- `sync_all()` runs on a **background worker thread** (typically a
  `ThreadPoolExecutor` with 1 worker). This means a slow or wedged provider
  can never stall the turn.
- Writes are serialized through a single worker so turn N lands before
  turn N+1. Providers don't need their own ordering guarantees.
- The worker should be lazily created on first use (most agents don't have
  an external memory provider) and shut down gracefully on session end.

### Handling interruptions

- **If the user interrupts the turn** (e.g., Ctrl+C, sends a new message
  before the response), `sync_all()` should NOT be called — the response
  the user saw is incomplete or non-existent, and writing it to the brain
  would pollute future recall with junk.
- `queue_prefetch_all()` for the next turn should also be skipped on
  interruption, since the next message is likely a retry of the same intent.

---

## 7. The Agent Instruction File: SOUL.md Integration

> **Key insight:** The brain-agent-loop has **two layers** — one automatic
> (the plugin), one intentional (the agent's own instructions). Both are
> necessary for the loop to compound.

### Two-Layer Architecture

| Layer | What | Who | Mechanism |
|-------|------|-----|-----------|
| **Automatic** | DETECT entities, READ brain before responding, WRITE timeline entries | Plugin (`GBrainMemoryProvider`) | `prefetch()` + `sync_turn()` — runs every turn without agent involvement |
| **Intentional** | Create link edges, save new pages, search deliberately, capture skills | Agent (the LLM itself) | Instructions in SOUL.md / AGENTS.md — agent decides when to act |

The plugin makes the brain **invisible** — context appears automatically,
timeline entries accumulate without any conscious action from the agent.

The agent instruction file makes the brain **deliberate** — the agent knows
it *has* a brain and can consciously choose to use its tools.

### What SOUL.md Contributes

```markdown
## Knowledge & Automatic GBrain Capture (MANDATORY)

You have a persistent knowledge brain (GBrain) connected via a memory
provider plugin. GBrain stores decisions, people, projects, entities,
and ideas across sessions.

**Primary save tool: `gbrain_put_page`.** Content is stored immediately
and searchable right away.

**Read tools:**
- `gbrain_search` — primary recall tool for most questions
- `gbrain_recall` — structured entity fact lookup

**Knowledge Graph Linking**
When the user states a relationship between entities in conversation,
call `gbrain_add_link` to create the appropriate edge.
Official link types: `works_at`, `founded`, `invested_in`, ...

**Things you don't yet know**
If the answer isn't in context, use web_search to find it. If what
you find is durable, save it to GBrain with `gbrain_put_page`.

**Pain-driven skill capture**
If a task requires 5+ tool calls or was harder than it should have
been, stop and save a skill before moving on.
```

### Why Both Layers Are Needed

| Without the plugin | Without the agent instructions |
|---|---|
| No automatic context injection. Agent only searches when it remembers to. | Agent has the tools but doesn't know they exist (or treats them as optional). |
| No timeline accumulation. Conversations don't compound. | No deliberate `gbrain_add_link` calls — graph never builds. |
| Every turn starts from zero. | No `gbrain_put_page` for new entities — brain stays stale. |

The plugin ensures the brain **compounds from conversation alone** — even
if the agent never consciously reaches for its tools. The instructions
ensure the agent can **deliberately extend the graph** with relationships
and new pages.

### Porting to Another Platform

If you're implementing the brain-agent-loop in a different agent framework,
you need both:

1. **Plugin/Middleware** — code that intercepts every turn to run prefetch
   and sync_turn automatically (the three-phase loop in sections 1–6).
2. **Agent personality file** — a `SOUL.md`, `AGENTS.md`, `.cursorrules`,
   or `CLAUDE.md` that tells the agent about the brain backend and instructs
   it to use `gbrain_*` tools deliberately.

Without the instruction file, the agent is a silent partner who doesn't
know it has memory. Without the plugin, the agent only remembers when
it consciously decides to — which means most conversations leave no trace.

---

## 8. Platform Migration Guide

### Porting to another framework

#### You need

1. A **brain backend** reachable via API (GBrain MCP server, custom API, etc.)
2. The **three interface methods** (`prefetch`, `sync_turn`, `queue_prefetch`)
3. A **memory manager** that orchestrates providers (or at least one provider)
4. An **agent loop** that calls prefetch before the LLM and sync after

#### Minimal implementation checklist

```
[ ] Entity detector function (regex/mask/NER)
[ ] Brain search function (wraps brain backend API)
[ ] prefetch(message) → context string or null
[ ] Injection point: append context to the current user message before LLM call
[ ] sync_turn(user_msg, assistant_msg) → void
[ ] Background dispatch: run sync_turn off the main thread
[ ] Interrupt guard: skip sync when turn was interrupted
[ ] Stop-word list (if using mask-based detection)
```

#### Language-specific notes

| Language | Detector approach | Background dispatch | Thread Safety |
|----------|-------------------|---------------------|---------------|
| Python | `str.split` + `set` lookup | `concurrent.futures.ThreadPoolExecutor` | `threading.RLock()` on MCP client |
| TypeScript | `String.split` + `Set.has` | `worker_threads` or `setTimeout(0)` chain | Mutex on HTTP client |
| Go | `strings.Fields` + `map[string]bool` | `goroutine` (trivially concurrent) | `sync.Mutex` on client |
| Rust | `split_whitespace` + `HashSet` | `tokio::spawn` | `Arc<Mutex<Client>>` |
| Ruby | `split` + `Set` | `Thread.new` | `Monitor`-guarded HTTP |

#### Customizing for your brain backend

If you're not using GBrain MCP, adapt the search and write calls:

| Concept | GBrain MCP call | Your equivalent |
|---------|----------------|-----------------|
| Search brain | `gbrain_search(name, limit=3)` | `POST /api/search?q={name}` |
| Write timeline | `add_timeline_entry({slug, date, summary, source})` | `POST /api/pages/{slug}/timeline` |
| Read page | `gbrain_get(slug)` | `GET /api/pages/{slug}` |

---

## 9. Verification

To confirm the loop is working:

1. **Mention a person the brain knows.** Ask "what do we know about {name}?"
   The agent should search the brain and return compiled truth, not hallucinate
   or do a web search.

2. **Discuss something new about a known entity.** Say "I heard Acme Corp
   just raised Series B." After the conversation, check the entity's brain
   page — it should have a new timeline entry with
   `source: "hermes-brain-loop"`.

3. **Ask about the same entity a day later.** The agent should pull brain
   context automatically, without you asking. If it doesn't reference the
   brain page, the loop isn't running.

4. **Monitor the logs.** Look for:
   ```
   brain-loop prefetch: N entities detected, M context blocks returned (candidates=[...])
   brain-loop: added timeline to /{slug}
   ```

5. **Inspect the injected context.** In the raw LLM API request, look for
   `<memory-context>` tags in the current user message. The fenced block
   contains the brain's compiled truth for detected entities.

---

## Appendix A: Stop-Word Mask

The reference implementation uses a set of 2,226 English stop words drawn
from common function words: articles, pronouns, prepositions, conjunctions,
verbs (be, have, do), adjectives (good, new, first, last), adverbs (very,
really, just), quantifiers, demonstratives, and discourse markers.

Words NOT in the stop list are treated as potential entity tokens and
combined into multi-word phrases until a stop word breaks the phrase.

Example:
```
Input:  "Tell me about Pedro and his work at SAS Institute"
Tokens: "Tell" "me" "about" "Pedro" "and" "his" "work" "at" "SAS" "Institute"
Mask:   STOP  STOP STOP  KEEP   STOP STOP  STOP  STOP KEEP  KEEP
                          └─── phrase break ──┘     └─────────┐
Candidates: ["Pedro", "SAS Institute"]                        │
                                   (stop "and" flushes Pedro) ┘
```

---

## Appendix B: Key Architecture Decisions (ADRs)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Same-turn sync prefetch, not deferred | Guide says "Read BEFORE responding." Deferred caching would lose context for the current turn. |
| 2 | Mask-based detection, no NER | Zero dependencies, µs latency. Can upgrade later. |
| 3 | No new entity creation in loop | Creating pages requires reasoning about notability — that's the agent's job, not the loop's. |
| 4 | Fire-and-forget for writes | sync_turn runs on background thread so a slow/wedged provider never blocks the turn. |
| 5 | Skip sync on interrupted turns | Writing incomplete responses to the brain pollutes future recall. |
| 6 | Provider isolation | Failures in prefetch/sync_turn never propagate. One broken provider can't break the loop. |
| 7 | Two-layer architecture (plugin + SOUL.md) | Automatic loop (plugin) handles DETECT/READ/WRITE. Agent instructions (SOUL.md) handle deliberate graph linking and page creation. Both required for compounding. |

---

*Derived from the [GBrain Brain-Agent Loop guide](https://github.com/garrytan/gbrain/blob/master/docs/guides/brain-agent-loop.md)
and the [Hermes GBrain Memory Provider Plugin](https://github.com/BrightCollage/hermes-gbrain-memory-plugin/tree/brain-agent-loop)
implementation. Part of the [GBrain Skillpack](https://github.com/garrytan/gbrain/blob/master/docs/GBRAIN_SKILLPACK.md).*
