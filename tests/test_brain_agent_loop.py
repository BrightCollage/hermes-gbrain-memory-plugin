"""Tests for the brain-agent-loop methods in the GBrain memory plugin.

Run with:
    python3 -m unittest tests/test_brain_agent_loop.py -v
"""
import json
import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock

# ── Mock Hermes core modules with real ABCs ──────────────────────────
import abc
class MemoryProvider(abc.ABC):
    """Mock of agent.memory_provider.MemoryProvider — real ABC so
    the plugin's method implementations aren't replaced by mocks."""
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""
    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass
    def sync_turn(self, user_content: str, assistant_content: str, *,
                session_id: str = "", messages=None) -> None:
        pass

class MockHermesConstants:
    @staticmethod
    def get_hermes_home() -> str:
        return "/tmp"

sys.modules["agent"] = MagicMock()
sys.modules["agent.memory_provider"] = MagicMock()
sys.modules["agent.memory_provider"].MemoryProvider = MemoryProvider
sys.modules["hermes_constants"] = MockHermesConstants()

# Import the plugin module
# Need to add both the plugin dir and the plugins dir to path so
# relative imports (._schemas) resolve correctly
_plugin_dir = os.path.join(os.path.dirname(__file__), "..")
_plugins_dir = os.path.dirname(_plugin_dir)
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)
if _plugins_dir not in sys.path:
    sys.path.insert(0, _plugins_dir)

# Can't use import __init__ because of relative imports; use importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "gbrain_plugin",
    os.path.join(_plugin_dir, "__init__.py"),
    submodule_search_locations=[],
)
plugin = importlib.util.module_from_spec(spec)
# Make relative imports work by registering as a package
plugin.__package__ = "gbrain_plugin"
sys.modules["gbrain_plugin"] = plugin
spec.loader.exec_module(plugin)


class MockMcpClient:
    """Simulates the GBrain MCP server for testing."""

    def __init__(self, seed_data=None):
        self.calls = []
        self.seed_data = seed_data or {
            "alice": [{
                "slug": "people/alice",
                "title": "Alice Example",
                "type": "person",
                "chunk_text": "Test contact used during plugin development.",
            }],
            "sas institute": [{
                "slug": "companies/sas-institute",
                "title": "SAS Institute",
                "type": "company",
                "chunk_text": "Software analytics company in Cary, North Carolina.",
            }],
            "racoony deployment": [{
                "slug": "projects/racoony",
                "title": "Racoony",
                "type": "project",
                "chunk_text": "Racoony Hermes Agent running on Kimori.",
            }],
            "kimori": [{
                "slug": "infrastructure/kimori-cluster",
                "title": "Kimori Cluster",
                "type": "infrastructure",
                "chunk_text": "Homelab K8s cluster on Talos.",
            }],
            "angie": [{
                "slug": "people/angie-chen",
                "title": "Angie Chen",
                "type": "person",
                "chunk_text": "Nuclear pharmacist at Cardinal Denver.",
            }],
        }

    def call(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self._respond(tool_name, arguments)

    def _respond(self, tool_name, arguments):
        if tool_name == "search":
            query = arguments.get("query", "").lower().strip()
            results = []
            for key, data in self.seed_data.items():
                if key in query or query in key:
                    results.extend(data)
            return {"content": [{"type": "text", "text": json.dumps(results)}]}
        if tool_name == "add_timeline_entry":
            return {"content": [{"type": "text", "text": json.dumps({"ok": True})}]}
        return {"content": []}


class TestExtractCandidates(unittest.TestCase):
    """Tests for _extract_candidates — the mask-based entity extraction."""

    def setUp(self):
        self.provider = plugin.GBrainMemoryProvider()

    def _extract(self, text):
        return self.provider._extract_candidates(text)

    def test_empty(self):
        self.assertEqual(self._extract(""), [])
        self.assertEqual(self._extract(None), [])

    def test_only_stop_words(self):
        self.assertEqual(self._extract("Who am I talking to right now?"), [])

    def test_kimori_query(self):
        result = self._extract("Can you check kimori to see if there's any stale resources?")
        self.assertEqual(result, ["kimori"])

    def test_alice_and_racoony(self):
        result = self._extract("Tell me about Alice and the Racoony deployment")
        self.assertEqual(result, ["Alice", "Racoony deployment"])

    def test_multi_entity(self):
        result = self._extract("What do you know about SAS Institute and Kimori?")
        self.assertEqual(result, ["SAS Institute", "Kimori"])

    def test_entity_after_preposition(self):
        result = self._extract("I talked to Alice at SAS Institute about the gbrain deploy on kimori")
        self.assertEqual(result, ["Alice", "SAS Institute", "gbrain deploy", "kimori"])

    def test_contraction_masked(self):
        result = self._extract("There's something about kimori")
        self.assertEqual(result, ["kimori"])

    def test_deployment(self):
        result = self._extract("Can you check all the deployment to see if they are working correctly on kimori?")
        self.assertEqual(result, ["deployment", "kimori"])


    def test_possessive_passes_through(self):
        """Possessives like 'Alice's' survive the mask (apostrophe preserved)."""
        result = self._extract("Alice's deployment on kimori")
        self.assertIn("Alice's", str(result))
        self.assertIn("kimori", result)


class TestSearchEntity(unittest.TestCase):
    """Tests for _search_entity — searching GBrain and formatting context."""

    def setUp(self):
        self.provider = plugin.GBrainMemoryProvider()
        self.provider._mcp = MockMcpClient()

    def test_known_entity_returns_context(self):
        result = self.provider._search_entity("Alice")
        self.assertIn("## Brain context", result)
        self.assertIn("Alice", result)

    def test_known_entity_with_type(self):
        result = self.provider._search_entity("SAS Institute")
        self.assertIn("company", result)
        self.assertIn("/companies/sas-institute", result)

    def test_nonexistent_entity_returns_empty(self):
        self.provider._mcp = MockMcpClient()
        result = self.provider._search_entity("xyzzy_nonexistent")
        self.assertEqual(result, "")

    def test_no_mcp_returns_empty(self):
        self.provider._mcp = None
        result = self.provider._search_entity("anything")
        self.assertEqual(result, "")


class TestResolveEntities(unittest.TestCase):
    """Tests for _resolve_entities — extracting and resolving to brain slugs."""

    def setUp(self):
        self.provider = plugin.GBrainMemoryProvider()
        self.provider._mcp = MockMcpClient()

    def test_person_and_company(self):
        entities = self.provider._resolve_entities("I talked to Alice from SAS Institute")
        slugs = {e["slug"] for e in entities}
        self.assertIn("people/alice", slugs)
        self.assertIn("companies/sas-institute", slugs)

    def test_skips_non_person_company(self):
        entities = self.provider._resolve_entities("Kimori cluster")
        for e in entities:
            self.assertNotEqual(e["slug"], "infrastructure/kimori-cluster")

    def test_no_entities_returns_empty(self):
        entities = self.provider._resolve_entities("Who am I?")
        self.assertEqual(entities, [])


class TestSyncTurn(unittest.TestCase):
    """Tests for sync_turn — the WRITE step."""

    def setUp(self):
        self.provider = plugin.GBrainMemoryProvider()
        self.mcp = MockMcpClient()
        self.provider._mcp = self.mcp

    def test_writes_timeline_for_detected_entities(self):
        self.provider.sync_turn(
            "I talked to Alice from SAS Institute today",
            "That's interesting!",
        )
        timeline_calls = [c for c in self.mcp.calls if c[0] == "add_timeline_entry"]
        slugs = {c[1]["slug"] for c in timeline_calls}
        self.assertIn("people/alice", slugs)
        self.assertIn("companies/sas-institute", slugs)

    def test_writes_nothing_for_empty_message(self):
        self.provider.sync_turn("", "response")
        timeline_calls = [c for c in self.mcp.calls if c[0] == "add_timeline_entry"]
        self.assertEqual(len(timeline_calls), 0)

    def test_writes_nothing_for_no_entities(self):
        self.provider.sync_turn("Who am I?", "You are Racoony.")
        timeline_calls = [c for c in self.mcp.calls if c[0] == "add_timeline_entry"]
        self.assertEqual(len(timeline_calls), 0)


class TestPrefetch(unittest.TestCase):
    """Tests for prefetch — the READ step."""

    def setUp(self):
        self.provider = plugin.GBrainMemoryProvider()
        self.mcp = MockMcpClient()
        self.provider._mcp = self.mcp

    def test_returns_context_for_known_entities(self):
        result = self.provider.prefetch("Tell me about Alice and SAS Institute")
        self.assertIn("## GBrain", result)
        self.assertIn("Alice", result)
        self.assertIn("SAS Institute", result)

    def test_returns_empty_for_no_entities(self):
        result = self.provider.prefetch("Who am I talking to right now?")
        self.assertEqual(result, "")

    def test_returns_empty_for_empty_query(self):
        result = self.provider.prefetch("")
        self.assertEqual(result, "")

    def test_no_mcp_returns_empty(self):
        self.provider._mcp = None
        result = self.provider.prefetch("Alice at SAS")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
