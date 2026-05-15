import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
except ImportError:
    print("Please install ragas and datasets: pip install ragas datasets")
    exit(1)

from graph.graph import app as research_app
from storage.vectorstore import get_vectorstore
from storage.embeddings import get_embeddings
from agents.orchestrator import get_llm

def load_test_queries(filepath="eval/test_queries.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def run_evaluation():
    queries = load_test_queries()
    
    # For demonstration, we evaluate just a random sample (e.g., 2 queries)
    # as evaluating all 20 via API or Local LLM can take a very long time.
    sample_queries = queries[:2]
    
    questions = []
    answers = []
    contexts = []
    
    print(f"🚀 Running benchmark on {len(sample_queries)} queries...")
    
    vectorstore = get_vectorstore()
    
    for i, q in enumerate(sample_queries, 1):
        print(f"\n[{i}/{len(sample_queries)}] Evaluating query: {q}")
        
        initial_state = {
            "query": q,
            "max_iterations": 2, # Keep iterations low for benchmarking speed
            "sub_tasks": [],
            "web_findings": [],
            "pdf_findings": [],
            "all_findings": [],
            "critic_score": 0.0,
            "critic_feedback": "",
            "critic_sufficient": False,
            "iteration_count": 0,
            "final_report": "",
            "sources": []
        }
        
        # 1. Run the LangGraph agent workflow
        start_time = time.time()
        result = research_app.invoke(initial_state)
        report = result.get("final_report", "No report generated.")
        end_time = time.time()
        
        print(f"   ⏱️ Graph Execution Time: {end_time - start_time:.2f} seconds")
        print(f"   🔄 Iterations taken: {result.get('iteration_count')}")
        
        # 2. Retrieve the context used (for Ragas metrics)
        docs = vectorstore.similarity_search(q, k=15)
        context_list = [doc.page_content for doc in docs]
        
        questions.append(q)
        answers.append(report)
        contexts.append(context_list)
        
    print("\n📊 Preparing RAGAS dataset...")
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    
    dataset = Dataset.from_dict(data)
    
    print("🧠 Running RAGAS metrics (Faithfulness & Answer Relevancy)...")
    try:
        # Note: RAGAS by default expects OPENAI_API_KEY. 
        # We attempt to override it with our project's LLM/Embeddings.
        # Depending on your 'ragas' library version, you may need to use wrappers.
        eval_llm = get_llm()
        eval_emb = get_embeddings()
        
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=eval_llm,
            embeddings=eval_emb
        )
        print("\n🏆 --- Benchmark Results ---")
        print(results)
    except Exception as e:
        print(f"\n❌ Evaluation failed. If you see authentication errors, RAGAS might be falling back to OpenAI.")
        print(f"Error Details: {e}")
        print("To fix: Set OPENAI_API_KEY in your .env file, OR check the ragas documentation for your specific local LLM wrapper.")

if __name__ == "__main__":
    run_evaluation()
