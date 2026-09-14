#!/usr/bin/env python3
"""FAISS vector search scale/latency benchmark.

The live Streamlit app only indexes ~20 schema/document chunks (fast to
build on every session start, which is what you want for an interactive
demo). To back up the "50,000+ embeddings / <120ms retrieval" claim in the
README with a real, reproducible number, this script builds a separate,
much larger FAISS index and measures actual query latency on *this*
machine.

Methodology note: vectors here are randomly generated at the same
dimensionality (384) that `all-MiniLM-L6-v2` produces, rather than real
sentence embeddings. FAISS's search latency for `IndexFlatL2` depends on the
number of vectors and their dimensionality, not on their semantic content,
so this measures genuine ANN search performance at scale without requiring
a network call to download the embedding model for every benchmark run
(useful in CI or offline environments). If you want an end-to-end benchmark
using real embeddings, pass --use-sentence-transformers (requires network
access to huggingface.co).

Usage:
    python scripts/benchmark_vector_search.py --num-vectors 50000 --num-queries 200
    python scripts/benchmark_vector_search.py --use-sentence-transformers --num-vectors 50000
"""
import argparse
import json
import time
import statistics

import numpy as np
import faiss


def build_index(num_vectors: int, dim: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    vectors = rng.rand(num_vectors, dim).astype("float32")
    index = faiss.IndexFlatL2(dim)
    start = time.perf_counter()
    index.add(vectors)
    build_ms = (time.perf_counter() - start) * 1000
    return index, vectors, build_ms


def run_benchmark(num_vectors: int, num_queries: int, dim: int, top_k: int, seed: int = 7):
    index, vectors, build_ms = build_index(num_vectors, dim)

    rng = np.random.RandomState(seed)
    queries = rng.rand(num_queries, dim).astype("float32")

    latencies_ms = []
    for i in range(num_queries):
        q = queries[i : i + 1]
        start = time.perf_counter()
        index.search(q, top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    result = {
        "num_vectors": num_vectors,
        "dimension": dim,
        "top_k": top_k,
        "num_queries": num_queries,
        "index_build_ms": round(build_ms, 2),
        "latency_ms": {
            "min": round(latencies_ms[0], 3),
            "p50": round(statistics.median(latencies_ms), 3),
            "p95": round(latencies_ms[int(0.95 * len(latencies_ms)) - 1], 3),
            "p99": round(latencies_ms[int(0.99 * len(latencies_ms)) - 1], 3),
            "max": round(latencies_ms[-1], 3),
            "mean": round(statistics.mean(latencies_ms), 3),
        },
    }
    return result


def run_benchmark_with_real_embeddings(num_vectors: int, num_queries: int, top_k: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    texts = [f"Synthetic enterprise document chunk number {i} about orders, customers, and inventory." for i in range(num_vectors)]
    start = time.perf_counter()
    vectors = model.encode(texts, show_progress_bar=False, batch_size=256).astype("float32")
    encode_ms = (time.perf_counter() - start) * 1000

    index = faiss.IndexFlatL2(dim)
    build_start = time.perf_counter()
    index.add(vectors)
    build_ms = (time.perf_counter() - build_start) * 1000

    query_texts = [f"find records related to topic {i}" for i in range(num_queries)]
    query_vectors = model.encode(query_texts, show_progress_bar=False).astype("float32")

    latencies_ms = []
    for i in range(num_queries):
        q = query_vectors[i : i + 1]
        start = time.perf_counter()
        index.search(q, top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    return {
        "num_vectors": num_vectors,
        "dimension": dim,
        "top_k": top_k,
        "num_queries": num_queries,
        "embedding_encode_ms_total": round(encode_ms, 2),
        "index_build_ms": round(build_ms, 2),
        "latency_ms": {
            "min": round(latencies_ms[0], 3),
            "p50": round(statistics.median(latencies_ms), 3),
            "p95": round(latencies_ms[int(0.95 * len(latencies_ms)) - 1], 3),
            "p99": round(latencies_ms[int(0.99 * len(latencies_ms)) - 1], 3),
            "max": round(latencies_ms[-1], 3),
            "mean": round(statistics.mean(latencies_ms), 3),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS vector search latency at scale.")
    parser.add_argument("--num-vectors", type=int, default=50000)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension (384 matches all-MiniLM-L6-v2)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-sentence-transformers", action="store_true",
                         help="Use real all-MiniLM-L6-v2 embeddings instead of random vectors (requires network access).")
    parser.add_argument("--output", default="scripts/benchmark_results.json")
    args = parser.parse_args()

    if args.use_sentence_transformers:
        result = run_benchmark_with_real_embeddings(args.num_vectors, args.num_queries, args.top_k)
        result["mode"] = "real_sentence_transformer_embeddings"
    else:
        result = run_benchmark(args.num_vectors, args.num_queries, args.dim, args.top_k)
        result["mode"] = "synthetic_random_vectors_matching_minilm_dimension"

    print(json.dumps(result, indent=2))

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
