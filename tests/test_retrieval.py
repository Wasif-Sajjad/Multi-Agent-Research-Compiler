import unittest
from unittest.mock import patch, MagicMock
from agents.synthesis import synthesis_node

class TestRetrieval(unittest.TestCase):
    
    @patch('agents.synthesis.get_vectorstore')
    @patch('agents.synthesis._call_synthesis_llm')
    def test_synthesis_node_retrieval(self, mock_llm, mock_get_vs):
        # Mock the vectorstore similarity search
        mock_vs = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "This is a retrieved semantic chunk."
        mock_doc.metadata = {"source": "https://example.com/test", "title": "Example Title"}
        
        # Return two identical documents to test source deduplication
        mock_vs.similarity_search.return_value = [mock_doc, mock_doc]
        mock_get_vs.return_value = mock_vs
        
        # Mock LLM generation output
        mock_llm.return_value = "# Final Report\n\nGenerated content based on [1]."
        
        state = {"query": "Test query"}
        result = synthesis_node(state)
        
        # Verify vector store was called with correct parameters
        mock_vs.similarity_search.assert_called_once_with("Test query", k=15)
        
        # Verify deduplication of sources
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0], "https://example.com/test")
        
        # Verify report generation
        self.assertIn("Final Report", result["final_report"])

if __name__ == '__main__':
    unittest.main()
