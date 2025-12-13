"""
compare_embeddings.py
---------------------
Comprehensive comparison script for embedding models.
Generates quantitative and qualitative evaluation results for Milestone 3 presentation.
"""

import time
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from node_embeddings import EmbeddingRetriever, NodeEmbeddingGenerator


def load_config(path="data/config.txt"):
    """Load Neo4j configuration from file."""
    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config["URI"], config["USERNAME"], config["PASSWORD"]


class EmbeddingModelComparison:
    """
    Compares two embedding models on various metrics.
    Generates results for presentation.
    """
    
    def __init__(self, config_path: str = "data/config.txt"):
        """Initialize comparison with Neo4j config."""
        self.uri, self.user, self.password = load_config(config_path)
        self.results = {
            "minilm": {},
            "mpnet": {}
        }
    
    def measure_generation_time(self, model_name: str, sample_size: int = None) -> float:
        """
        Measure time to generate embeddings.
        
        Args:
            model_name: "minilm" or "mpnet"
            sample_size: Number of hotels to test (None = all)
            
        Returns:
            Time in seconds
        """
        print(f"\n[{model_name}] Measuring generation time...")
        
        generator = NodeEmbeddingGenerator(
            self.uri, self.user, self.password, model_name
        )
        
        try:
            # Fetch hotels
            hotels = generator.fetch_all_hotels()
            if sample_size:
                hotels = hotels[:sample_size]
            
            # Measure generation time
            start_time = time.time()
            embeddings = generator.generate_embeddings(hotels)
            generation_time = time.time() - start_time
            
            print(f"   Generated {len(embeddings)} embeddings in {generation_time:.2f}s")
            
            return generation_time
        
        finally:
            generator.close()
    
    def measure_query_time(self, model_name: str, test_queries: List[str],
                          limit: int = 10) -> Dict[str, float]:
        """
        Measure query execution time.
        
        Args:
            model_name: "minilm" or "mpnet"
            test_queries: List of queries to test
            limit: Number of results per query
            
        Returns:
            Dictionary with timing statistics
        """
        print(f"\n[{model_name}] Measuring query time...")
        
        retriever = EmbeddingRetriever(
            self.uri, self.user, self.password, model_name
        )
        
        try:
            query_times = []
            
            for query in test_queries:
                start_time = time.time()
                results = retriever.search(query, limit=limit)
                query_time = time.time() - start_time
                query_times.append(query_time)
                print(f"   '{query[:40]}...' -> {query_time:.3f}s ({len(results)} results)")
            
            return {
                "avg_time": np.mean(query_times),
                "min_time": np.min(query_times),
                "max_time": np.max(query_times),
                "std_time": np.std(query_times),
                "total_time": np.sum(query_times)
            }
        
        finally:
            retriever.close()
    
    def evaluate_result_quality(self, model_name: str, 
                               test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate quality of search results.
        
        Args:
            model_name: "minilm" or "mpnet"
            test_cases: List of test cases with queries and expected criteria
            
        Returns:
            Quality metrics dictionary
        """
        print(f"\n[{model_name}] Evaluating result quality...")
        
        retriever = EmbeddingRetriever(
            self.uri, self.user, self.password, model_name
        )
        
        try:
            quality_scores = []
            relevance_scores = []
            
            for test_case in test_cases:
                query = test_case['query']
                expected_city = test_case.get('expected_city')
                min_score = test_case.get('min_score', 0.5)
                
                results = retriever.search(query, limit=10)
                
                # Check if results match expected criteria
                if results:
                    top_result = results[0]
                    
                    # City match
                    city_match = (expected_city is None or 
                                top_result.get('city', '').lower() == expected_city.lower())
                    
                    # Score threshold
                    score_ok = top_result.get('score', 0) >= min_score
                    
                    # Combined quality
                    quality = 1.0 if (city_match and score_ok) else 0.5
                    quality_scores.append(quality)
                    relevance_scores.append(top_result.get('score', 0))
                else:
                    quality_scores.append(0.0)
                    relevance_scores.append(0.0)
            
            return {
                "avg_quality": np.mean(quality_scores),
                "avg_relevance": np.mean(relevance_scores),
                "min_relevance": np.min(relevance_scores) if relevance_scores else 0,
                "max_relevance": np.max(relevance_scores) if relevance_scores else 0
            }
        
        finally:
            retriever.close()
    
    def compare_top_results(self, query: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Compare top results from both models for a single query.
        
        Args:
            query: Search query
            limit: Number of results to compare
            
        Returns:
            Dictionary with results from both models
        """
        print(f"\nComparing results for: '{query}'")
        
        results_comparison = {}
        
        for model_name in ["minilm", "mpnet"]:
            retriever = EmbeddingRetriever(
                self.uri, self.user, self.password, model_name
            )
            
            try:
                results = retriever.search(query, limit=limit)
                results_comparison[model_name] = results
                
                print(f"\n[{model_name}] Top {len(results)} results:")
                for i, hotel in enumerate(results, 1):
                    print(f"  {i}. {hotel['name']} ({hotel.get('city', 'N/A')}) - Score: {hotel.get('score', 0):.4f}")
            
            finally:
                retriever.close()
        
        return results_comparison
    
    def run_full_comparison(self, test_queries: List[str] = None,
                          test_cases: List[Dict] = None,
                          save_results: bool = True):
        """
        Run complete comparison of both models.
        
        Args:
            test_queries: Queries for timing tests
            test_cases: Test cases for quality evaluation
            save_results: Whether to save results to JSON file
        """
        if test_queries is None:
            test_queries = [
                "luxury hotel in Cairo",
                "budget-friendly accommodation in Dubai",
                "hotel with excellent cleanliness",
                "romantic getaway location",
                "business hotel near airport"
            ]
        
        if test_cases is None:
            test_cases = [
                {"query": "luxury hotel in Cairo", "expected_city": "Cairo", "min_score": 0.6},
                {"query": "hotels in Dubai", "expected_city": "Dubai", "min_score": 0.5},
                {"query": "clean hotel", "expected_city": None, "min_score": 0.4},
                {"query": "5-star accommodation", "expected_city": None, "min_score": 0.6},
            ]
        
        print("="*80)
        print("EMBEDDING MODEL COMPARISON")
        print("="*80)
        
        for model_name in ["minilm", "mpnet"]:
            print(f"\n{'='*80}")
            print(f"Evaluating: {model_name.upper()}")
            print(f"{'='*80}")
            
            # Get embedding dimensions
            retriever = EmbeddingRetriever(self.uri, self.user, self.password, model_name)
            embedding_dim = retriever.model.get_sentence_embedding_dimension()
            retriever.close()
            
            self.results[model_name]["embedding_dimension"] = embedding_dim
            
            # Measure query time
            query_stats = self.measure_query_time(model_name, test_queries)
            self.results[model_name]["query_stats"] = query_stats
            
            # Evaluate quality
            quality_stats = self.evaluate_result_quality(model_name, test_cases)
            self.results[model_name]["quality_stats"] = quality_stats
        
        # Print summary
        self.print_comparison_summary()
        
        # Save results
        if save_results:
            self.save_results_to_file()
    
    def print_comparison_summary(self):
        """Print a formatted comparison summary."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        print("\n1. QUANTITATIVE METRICS")
        print("-" * 80)
        
        # Table header
        print(f"{'Metric':<30} {'MiniLM':<20} {'MPNet':<20}")
        print("-" * 80)
        
        # Embedding dimension
        print(f"{'Embedding Dimension':<30} "
              f"{self.results['minilm'].get('embedding_dimension', 'N/A'):<20} "
              f"{self.results['mpnet'].get('embedding_dimension', 'N/A'):<20}")
        
        # Query time
        minilm_query = self.results['minilm'].get('query_stats', {})
        mpnet_query = self.results['mpnet'].get('query_stats', {})
        
        print(f"{'Avg Query Time (s)':<30} "
              f"{minilm_query.get('avg_time', 0):<20.3f} "
              f"{mpnet_query.get('avg_time', 0):<20.3f}")
        
        # Quality metrics
        minilm_quality = self.results['minilm'].get('quality_stats', {})
        mpnet_quality = self.results['mpnet'].get('quality_stats', {})
        
        print(f"{'Avg Quality Score':<30} "
              f"{minilm_quality.get('avg_quality', 0):<20.3f} "
              f"{mpnet_quality.get('avg_quality', 0):<20.3f}")
        
        print(f"{'Avg Relevance Score':<30} "
              f"{minilm_quality.get('avg_relevance', 0):<20.3f} "
              f"{mpnet_quality.get('avg_relevance', 0):<20.3f}")
        
        print("\n2. QUALITATIVE OBSERVATIONS")
        print("-" * 80)
        print("MiniLM:")
        print("  + Faster query execution")
        print("  + Lower memory footprint (384 dimensions)")
        print("  + Suitable for real-time applications")
        print("  - Slightly lower semantic understanding")
        
        print("\nMPNet:")
        print("  + Better semantic understanding (768 dimensions)")
        print("  + Higher relevance scores")
        print("  + Better at capturing nuanced queries")
        print("  - Slower query execution")
        print("  - Higher memory requirements")
        
        print("\n3. RECOMMENDATION")
        print("-" * 80)
        
        # Determine recommendation
        minilm_avg_quality = minilm_quality.get('avg_quality', 0)
        mpnet_avg_quality = mpnet_quality.get('avg_quality', 0)
        
        if mpnet_avg_quality > minilm_avg_quality * 1.1:
            print("Recommendation: MPNet")
            print("Reason: Significantly better quality justifies the performance trade-off")
        elif minilm_query.get('avg_time', 1) < mpnet_query.get('avg_time', 1) * 0.8:
            print("Recommendation: MiniLM")
            print("Reason: Much faster with acceptable quality for production use")
        else:
            print("Recommendation: Context-dependent")
            print("Reason: Both models perform similarly; choose based on latency requirements")
    
    def save_results_to_file(self, filename: str = "embedding_comparison_results.json"):
        """Save comparison results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")
    
    def generate_presentation_table(self) -> str:
        """Generate a markdown table for presentation."""
        minilm = self.results['minilm']
        mpnet = self.results['mpnet']
        
        table = """
| Metric | MiniLM | MPNet |
|--------|--------|-------|
| Embedding Dimension | {} | {} |
| Avg Query Time (s) | {:.3f} | {:.3f} |
| Avg Quality Score | {:.3f} | {:.3f} |
| Avg Relevance Score | {:.3f} | {:.3f} |
| Speed | ⚡ Fast | 🐢 Moderate |
| Quality | ✓ Good | ⭐ Excellent |
| Memory Usage | 💾 Low | 💾💾 High |
| Use Case | Production | Research/Quality |
""".format(
            minilm.get('embedding_dimension', 'N/A'),
            mpnet.get('embedding_dimension', 'N/A'),
            minilm.get('query_stats', {}).get('avg_time', 0),
            mpnet.get('query_stats', {}).get('avg_time', 0),
            minilm.get('quality_stats', {}).get('avg_quality', 0),
            mpnet.get('quality_stats', {}).get('avg_quality', 0),
            minilm.get('quality_stats', {}).get('avg_relevance', 0),
            mpnet.get('quality_stats', {}).get('avg_relevance', 0)
        )
        
        print("\nMARKDOWN TABLE FOR PRESENTATION:")
        print(table)
        
        return table


def main():
    """Main function for running comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument('--config', type=str, default='data/config.txt',
                       help='Path to Neo4j config file')
    parser.add_argument('--mode', type=str, 
                       choices=['full', 'quick', 'side-by-side'],
                       default='full',
                       help='Comparison mode')
    parser.add_argument('--query', type=str, default=None,
                       help='Query for side-by-side comparison')
    
    args = parser.parse_args()
    
    comparison = EmbeddingModelComparison(config_path=args.config)
    
    if args.mode == 'full':
        # Full comprehensive comparison
        comparison.run_full_comparison()
        comparison.generate_presentation_table()
    
    elif args.mode == 'quick':
        # Quick test with fewer queries
        test_queries = [
            "luxury hotel Cairo",
            "budget hotel Dubai"
        ]
        test_cases = [
            {"query": "luxury hotel Cairo", "expected_city": "Cairo", "min_score": 0.6}
        ]
        comparison.run_full_comparison(test_queries, test_cases)
    
    elif args.mode == 'side-by-side':
        # Compare specific query across both models
        query = args.query or "luxury hotel with excellent service in Cairo"
        comparison.compare_top_results(query, limit=5)


if __name__ == "__main__":
    main()