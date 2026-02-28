"""
Benchmark script — evaluates the local RAG and reranker pipelines.

Metrics:
  1. Retrieval latency / throughput (bi-encoder + FAISS)
  2. Relevance (NDCG@10) on a real MS MARCO dev-small slice
  3. End-to-end generation latency

Requires: pip install datasets scikit-learn
"""
import time
import numpy as np
from sklearn.metrics import ndcg_score
from rag import RAGPipeline
from reranker import CrossEncoderReranker


def _load_msmarco_slice(n_queries: int = 20):
    """Load a small slice of MS MARCO passage ranking dev set."""
    from datasets import load_dataset

    print(f"Downloading MS MARCO dev-small slice ({n_queries} queries)...")
    ds = load_dataset(
        "microsoft/ms_marco", "v1.1",
        split=f"validation[:{n_queries}]",
        trust_remote_code=True,
    )
    return ds


def run_benchmarks():
    print("=" * 60)
    print("  RAG + Reranker Benchmark Suite")
    print("=" * 60)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading models...")
    t0 = time.time()
    rag = RAGPipeline()
    reranker = CrossEncoderReranker()
    print(f"      Models loaded in {time.time() - t0:.1f}s\n")

    # ── Retrieval latency on synthetic corpus ────────────────────────────────
    print("[2/4] Retrieval latency & throughput (synthetic corpus)")
    corpus = [
        {"url": "doc1", "content": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."},
        {"url": "doc2", "content": "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor."},
        {"url": "doc3", "content": "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas."},
        {"url": "doc4", "content": "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states."},
        {"url": "doc5", "content": "Paris is the capital and most populous city of France. The city is a major transport hub."},
        {"url": "doc6", "content": "The speed of light in vacuum is exactly 299,792,458 metres per second."},
    ]
    rag.build_index(corpus)

    queries = [
        "What is the capital of France?",
        "How fast does light travel?",
        "Where is the Statue of Liberty?",
        "Tell me about the Eiffel Tower.",
    ]

    latencies = []
    for _ in range(10):
        for q in queries:
            t0 = time.time()
            rag.retrieve(q, top_k=3)
            latencies.append(time.time() - t0)

    avg_ms = np.mean(latencies) * 1000
    throughput = 1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0
    print(f"      Avg retrieval latency : {avg_ms:.2f} ms")
    print(f"      Throughput            : {throughput:.0f} queries/sec\n")

    # ── NDCG on MS MARCO ─────────────────────────────────────────────────────
    print("[3/4] Relevance (NDCG@10) on MS MARCO dev-small")
    try:
        ds = _load_msmarco_slice(n_queries=20)

        ndcg_scores = []
        evaluated = 0
        for row in ds:
            query = row["query"]
            passages = row.get("passages", {})
            texts = passages.get("passage_text", [])
            labels = passages.get("is_selected", [])

            if not texts or not labels or len(texts) < 2:
                continue

            # Build a small FAISS index from these passages
            docs = [{"url": f"p{i}", "content": t} for i, t in enumerate(texts)]
            rag.build_index(docs)
            retrieved = rag.retrieve(query, top_k=min(10, len(texts)))

            if len(retrieved) < 2:
                continue

            # Build score vectors aligned to original passage order
            bi_scores = [0.0] * len(texts)
            for r in retrieved:
                # Find which original passage this chunk came from
                for i, t in enumerate(texts):
                    if r["text"] in t or t in r["text"]:
                        bi_scores[i] = max(bi_scores[i], r["score"])
                        break

            true_labels = [float(l) for l in labels]

            if sum(true_labels) == 0:
                continue

            try:
                score = ndcg_score([true_labels], [bi_scores])
                ndcg_scores.append(score)
                evaluated += 1
            except Exception:
                continue

        if ndcg_scores:
            print(f"      Evaluated queries : {evaluated}")
            print(f"      Mean NDCG@10     : {np.mean(ndcg_scores):.4f}")
            print(f"      Median NDCG@10   : {np.median(ndcg_scores):.4f}\n")
        else:
            print("      Could not compute NDCG (no valid queries).\n")

    except Exception as e:
        print(f"      MS MARCO download failed: {e}")
        print("      Falling back to synthetic NDCG...\n")

        q = "What are the major landmarks in Paris?"
        retrieved = rag.retrieve(q, top_k=5)
        bi_scores = [r["score"] for r in retrieved]
        texts = [r["text"] for r in retrieved]
        reranked = reranker.rerank(q, texts)
        cross_scores = [dict(reranked).get(t, 0.0) for t in texts]
        if len(bi_scores) > 1:
            ndcg = ndcg_score([cross_scores], [bi_scores])
            print(f"      Synthetic NDCG: {ndcg:.4f}\n")

    # ── E2E generation latency ───────────────────────────────────────────────
    print("[4/4] End-to-end generation latency")
    e2e_times = []
    for q in ["How fast does light travel?"]:
        t0 = time.time()
        retrieved = rag.retrieve(q, top_k=2)
        reranked = reranker.rerank(q, [r["text"] for r in retrieved])
        context = "\n".join([r[0] for r in reranked[:2]])
        answer = rag.generate_answer(q, context)
        e2e_times.append(time.time() - t0)

    print(f"      Avg E2E latency: {np.mean(e2e_times):.2f}s")
    print(f"\n{'=' * 60}")
    print("  Benchmark complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_benchmarks()
