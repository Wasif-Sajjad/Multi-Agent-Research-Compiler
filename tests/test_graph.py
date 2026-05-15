import unittest
from graph.router import dispatch_agents, aggregator_node, quality_router

class TestGraphRouting(unittest.TestCase):
    
    def test_dispatch_agents_both(self):
        state = {
            "sub_tasks": [
                {"question": "Q1", "source_type": "web"},
                {"question": "Q2", "source_type": "pdf"}
            ]
        }
        sends = dispatch_agents(state)
        self.assertEqual(len(sends), 2)
        nodes = [s.node for s in sends]
        self.assertIn("web_search_node", nodes)
        self.assertIn("pdf_reader_node", nodes)
        
    def test_dispatch_agents_fallback(self):
        # Even if subtasks are empty, it should fallback to web search safely
        state = {"sub_tasks": []}
        sends = dispatch_agents(state)
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "web_search_node")

    def test_aggregator_node(self):
        state = {
            "web_findings": [{"id": 1}],
            "pdf_findings": [{"id": 2}],
            "iteration_count": 1
        }
        result = aggregator_node(state)
        # Should combine both arrays into all_findings
        self.assertEqual(len(result["all_findings"]), 2)
        # Should increment iteration count
        self.assertEqual(result["iteration_count"], 2)

    def test_quality_router(self):
        # Success path
        state_pass = {"critic_sufficient": True, "iteration_count": 1, "max_iterations": 3}
        self.assertEqual(quality_router(state_pass), "synthesis_node")
        
        # Iteration hard cap path
        state_cap = {"critic_sufficient": False, "iteration_count": 3, "max_iterations": 3}
        self.assertEqual(quality_router(state_cap), "synthesis_node")
        
        # Loop path (rejected but under iteration cap)
        state_loop = {"critic_sufficient": False, "iteration_count": 1, "max_iterations": 3}
        self.assertEqual(quality_router(state_loop), "orchestrator_node")

if __name__ == '__main__':
    unittest.main()
