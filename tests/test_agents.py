import unittest
from unittest.mock import patch
from graph.state import ResearchState
from agents.orchestrator import orchestrator_node
from agents.critic import critic_node

class TestAgents(unittest.TestCase):
    
    @patch('agents.orchestrator._call_llm')
    def test_orchestrator_node(self, mock_call_llm):
        # Mock LLM returning a valid JSON list
        mock_call_llm.return_value = '''
        [
            {"question": "What is LangGraph?", "source_type": "web", "priority": 1},
            {"question": "Extract from manual", "source_type": "pdf", "priority": 2}
        ]
        '''
        state = {"query": "Test query"}
        result = orchestrator_node(state)
        
        self.assertIn("sub_tasks", result)
        self.assertEqual(len(result["sub_tasks"]), 2)
        self.assertEqual(result["sub_tasks"][0]["source_type"], "web")

    @patch('agents.critic._call_critic_llm')
    def test_critic_node_sufficient(self, mock_critic_llm):
        # Mock Critic LLM returning valid JSON passing the threshold
        mock_critic_llm.return_value = '''
        {
            "score": 0.8,
            "sufficient": true,
            "gaps": [],
            "feedback": "Looks good."
        }
        '''
        state = {
            "query": "Test query",
            "all_findings": [{"title": "Test", "content": "Content", "source_type": "web"}]
        }
        result = critic_node(state)
        
        self.assertIn("critic_score", result)
        self.assertEqual(result["critic_score"], 0.8)
        self.assertTrue(result["critic_sufficient"])

    @patch('agents.critic._call_critic_llm')
    def test_critic_node_insufficient(self, mock_critic_llm):
        # Mock Critic LLM returning a failing score
        mock_critic_llm.return_value = '''
        {
            "score": 0.5,
            "sufficient": false,
            "gaps": ["Missing recent data", "Only one source used"],
            "feedback": "Need more diverse info."
        }
        '''
        state = {
            "query": "Test query",
            "all_findings": [{"title": "Test", "content": "Content", "source_type": "web"}]
        }
        result = critic_node(state)
        
        self.assertFalse(result["critic_sufficient"])
        self.assertIn("Missing recent data", result["critic_feedback"])

if __name__ == '__main__':
    unittest.main()
