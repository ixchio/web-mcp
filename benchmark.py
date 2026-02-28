"""
Benchmark script for evaluating the local RAG and reranker pipelines.
Calculates Latency, Throughput, and NDCG using cross-encoder scores as proxy ground-truth.
"""
import time
import numpy as np
from sklearn.metrics import ndcg_score
from rag import RAGPipeline
from reranker import CrossEncoderReranker

def run_benchmarks():
    print("Loading models for benchmarking (this might take a few moments)...")
    start_time = time.time()
    rag = RAGPipeline()
    reranker = CrossEncoderReranker()
    print(f"Models successfully loaded in {time.time() - start_time:.2f}s.\n")

    corpus = [
        {"url": "doc1", "content": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."},
        {"url": "doc2", "content": "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor. It was a gift from France."},
        {"url": "doc3", "content": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas."},
        {"url": "doc4", "content": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states."},
        {"url": "doc5", "content": "Paris is the capital and most populous city of France. The city is a major railway, highway, and air-transport hub."},
        {"url": "doc6", "content": "The speed of light in vacuum is exactly 299,792,458 metres per second."},
    ]

    print("Building FAISS index on test corpus...")
    idx_start = time.time()
    rag.build_index(corpus)
    print(f"Index built in {time.time() - idx_start:.3f}s with {rag.index.ntotal} chunks.\n")

    queries = [
        "What is the capital of France?",
        "How fast does light travel?",
        "Where is the Statue of Liberty?",
        "Tell me about the Eiffel Tower."
    ]

    # --- 1. Retrieval Latency & Throughput ---
    print("--- 1. Retrieval Only Benchmarks ---")
    latencies = []
    num_runs = 10
    for _ in range(num_runs):
        for q in queries:
            t0 = time.time()
            res = rag.retrieve(q, top_k=3)
            latencies.append(time.time() - t0)
    
    avg_latency = np.mean(latencies)
    throughput = 1.0 / avg_latency if avg_latency > 0 else 0
    print(f"Average Retrieval Latency (Bi-encoder): {avg_latency*1000:.2f} ms")
    print(f"Retrieval Throughput: {throughput:.2f} queries/sec\n")

    # --- 2. Relevance (NDCG) ---
    print("--- 2. Relevance (NDCG) Benchmarks ---")
    # Simulate NDCG by using the exact Cross-Encoder score as True Relevance
    # to evaluate how good the base Bi-Encoder FAISS retrieval ranking is.
    q = "What are the major landmarks in Paris?"
    
    # Get top 5 from FAISS (bi-encoder)
    retrieved = rag.retrieve(q, top_k=5)
    bi_scores = [r["score"] for r in retrieved]
    texts = [r["text"] for r in retrieved]
    
    # Score them with cross-encoder
    reranked = reranker.rerank(q, texts)
    text_to_cross_score = {t: s for t, s in reranked}
    cross_scores = [text_to_cross_score[t] for t in texts]

    if len(bi_scores) > 1:
        try:
            # We want to see how the predicted order (bi_scores) compares against the true relevance (cross_scores)
            ndcg = ndcg_score([cross_scores], [bi_scores])
            print(f"NDCG corresponding to Bi-encoder order vs Cross-encoder truth: {ndcg:.4f}\n")
        except Exception as e:
            print(f"NDCG calculation error: {e}\n")
    else:
        print("Not enough results for NDCG.\n")

    # --- 3. Full E2E Latency ---
    print("--- 3. End-to-End Generation Benchmarks ---")
    e2e_latencies = []
    # Generation test takes longer, so we only do a couple queries
    for q in ["How fast does light travel?"]:
        t0 = time.time()
        retrieved = rag.retrieve(q, top_k=2)
        # rerank
        reranked = reranker.rerank(q, [r["text"] for r in retrieved])
        best_context = "\n".join([r[0] for r in reranked[:2]])
        # generate
        ans = rag.generate_answer(q, best_context)
        e2e_latencies.append(time.time() - t0)
    
    print(f"Average E2E Latency (Search -> Rerank -> Generate): {np.mean(e2e_latencies):.2f}s")
    print("\nBenchmark tests completed.")

if __name__ == "__main__":
    run_benchmarks()
