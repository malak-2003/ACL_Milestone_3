"""
Embedding Model Comparison Script
---------------------------------
Compares MiniLM vs MPNet embeddings on identical hotel search queries.

Outputs:
- Top-k retrieved hotels per model
- Cosine similarity scores
- Overlap between results
- Simple evaluation metrics

Designed for Milestone 3 evaluation.
"""

import json
import numpy as np
from typing import List, Dict
from pathlib import Path

from embeddings_retreiver import EmbeddingRetriever
from Create_kg import load_config


class EmbeddingModelComparator:
    def __init__(self, config_path: str):
        uri, user, password = load_config(config_path)

        self.minilm = EmbeddingRetriever(uri, user, password, "minilm")
        self.mpnet = EmbeddingRetriever(uri, user, password, "mpnet")

    def close(self):
        self.minilm.close()
        self.mpnet.close()

    def compare_single_query(
        self,
        query: str,
        limit: int = 5,
        city_filter: str = None
    ) -> Dict:
        """
        Compare MiniLM and MPNet on a single query.
        """

        print(f"\nComparing models for query: '{query}'")

        minilm_results = self.minilm.search(
            query, limit=limit, city_filter=city_filter
        )

        mpnet_results = self.mpnet.search(
            query, limit=limit, city_filter=city_filter
        )

        def simplify(results):
            return [
                {
                    "hotel_id": r.get("hotel_id"),
                    "name": r.get("name"),
                    "city": r.get("city"),
                    "score": round(r.get("score", 0), 4),
                }
                for r in results
            ]

        minilm_simple = simplify(minilm_results)
        mpnet_simple = simplify(mpnet_results)

        # Overlap analysis
        minilm_ids = {r["hotel_id"] for r in minilm_simple}
        mpnet_ids = {r["hotel_id"] for r in mpnet_simple}

        overlap = list(minilm_ids.intersection(mpnet_ids))

        comparison = {
            "query": query,
            "minilm": minilm_simple,
            "mpnet": mpnet_simple,
            "overlap_hotel_ids": overlap,
            "overlap_count": len(overlap),
        }

        return comparison

    def run_batch_comparison(
        self,
        queries: List[str],
        limit: int = 5,
        save_path: str = "embedding_comparison_results.json"
    ):
        """
        Run comparison on multiple queries and save results.
        """

        all_results = []

        for q in queries:
            result = self.compare_single_query(q, limit=limit)
            all_results.append(result)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Comparison results saved to {save_path}")
        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare MiniLM and MPNet embedding models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/config.txt",
        help="Path to Neo4j config file"
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        required=True,
        help="Queries to evaluate"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top-k results to compare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embedding_comparison_results.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    comparator = EmbeddingModelComparator(args.config)
    try:
        comparator.run_batch_comparison(
            args.queries,
            limit=args.limit,
            save_path=args.output
        )
    finally:
        comparator.close()


if __name__ == "__main__":
    main()
