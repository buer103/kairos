"""Quick smoke test for Kairos modules."""
from kairos.core.state import Case
from kairos.prompt.template import PromptBuilder
from kairos.tools.registry import get_all_tools
from kairos.middleware import EvidenceTracker, ConfidenceScorer, ContextCompressor
from kairos.infra.rag.vector_store import VectorStore
from kairos.infra.knowledge.store import KnowledgeSchema, KnowledgeStore
from kairos.infra.evidence.tracker import EvidenceDB
from kairos.agents.types import BUILTIN_TYPES
from kairos.core.middleware import MiddlewarePipeline
import kairos.tools.rag_search
import kairos.tools.knowledge_lookup
import kairos.agents.factory

def test_all():
    # Tools
    tools = list(get_all_tools().keys())
    assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"
    print(f"  ✅ Tools: {tools}")

    # Sub-Agent types
    assert len(BUILTIN_TYPES) == 3
    print(f"  ✅ Sub-Agent types: {list(BUILTIN_TYPES.keys())}")

    # PromptBuilder — all 3 modes
    for mode in ['default', 'diagnostic', 'minimal']:
        pb = PromptBuilder(mode=mode, agent_name='Test', soul='Be helpful.')
        prompt = pb.build()
        assert len(prompt) > 100
    print(f"  ✅ PromptBuilder: 3 modes working")

    # VectorStore
    vs = VectorStore()
    vs.add(['Document about AI agents', 'Document about cars', 'Weather report'])
    r = vs.search('AI agents', 3)
    assert len(r) >= 1
    print(f"  ✅ VectorStore: {len(r)} results, top: '{r[0]['content'][:30]}...'")

    # KnowledgeStore
    class FD(KnowledgeSchema):
        signal_name: str = ''
        root_cause: str = ''

    ks = KnowledgeStore(schema=FD)
    ks.insert(FD(id='1', signal_name='temp', root_cause='overheat'))
    ks.insert(FD(id='2', signal_name='volt', root_cause='short'))
    results = ks.query({'root_cause': 'overheat'})
    assert len(results) == 1
    assert results[0].signal_name == 'temp'
    print(f"  ✅ KnowledgeStore: query returned {len(results)} result")

    # EvidenceDB
    case = Case(id='test-42')
    case.add_step('read_file', {'path': '/tmp/a.txt'})
    db = EvidenceDB()
    path = db.save(case)
    loaded = db.load('test-42')
    assert loaded is not None
    assert len(loaded.steps) == 1
    print(f"  ✅ EvidenceDB: saved to {path.name}, loaded {len(loaded.steps)} step")

    # Middleware chain
    pipeline = MiddlewarePipeline([
        EvidenceTracker(),
        ConfidenceScorer(),
        ContextCompressor(),
    ])
    print(f"  ✅ MiddlewarePipeline: 3 layers composed")

    print()
    print("🎉 ALL SMOKE TESTS PASSED")

if __name__ == "__main__":
    test_all()
