"""
WEEK 4: VECTOR SEARCH TUNING
Optimize vector search performance, tune parameters, and benchmark different strategies
"""

import time
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

from data_structures_chromadb import DataStructuresKB


@dataclass
class SearchBenchmark:
    """Benchmark result for a search operation"""
    strategy: str
    query: str
    num_results: int
    response_time_ms: float
    recall: float
    precision: float
    ndcg: float
    relevant_docs_found: int
    total_relevant_docs: int
    timestamp: str


class VectorSearchTuner:
    """
    Optimize and tune vector search performance
    Includes benchmarking, parameter tuning, and performance evaluation
    """
    
    def __init__(self, kb: DataStructuresKB = None):
        """Initialize search tuner"""
        self.kb = kb or DataStructuresKB()
        self.benchmarks: List[SearchBenchmark] = []
        self.tuning_results: Dict = {}
        
        print("✓ Vector Search Tuner initialized")
    
    # ==================== PARAMETER TUNING ====================
    
    def tune_search_parameters(self,
                              query: str,
                              n_results_values: List[int] = None,
                              test_queries: Optional[List[str]] = None) -> Dict:
        """
        Tune different search parameters and find optimal configuration
        """
        if n_results_values is None:
            n_results_values = [1, 3, 5, 10, 20]
        
        if test_queries is None:
            test_queries = [query]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "parameter_variations": {}
        }
        
        for n_results in n_results_values:
            print(f"\nTesting with n_results={n_results}")
            
            timing_results = []
            quality_scores = []
            
            for test_query in test_queries:
                start = time.time()
                search_result = self.kb.search(test_query, n_results=n_results)
                elapsed = time.time() - start
                
                timing_results.append(elapsed * 1000)  # Convert to ms
                
                # Quality score (similarity average)
                if search_result["results"]:
                    quality = np.mean([r['similarity_score'] 
                                      for r in search_result["results"]])
                    quality_scores.append(quality)
            
            avg_time = np.mean(timing_results)
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            results["parameter_variations"][f"n_results={n_results}"] = {
                "avg_response_time_ms": round(avg_time, 2),
                "avg_quality_score": round(avg_quality, 4),
                "queries_tested": len(test_queries),
                "optimal": avg_quality > 0.7 and avg_time < 100  # Heuristic
            }
            
            print(f"  Avg Time: {avg_time:.2f}ms | Quality: {avg_quality:.4f}")
        
        self.tuning_results["parameter_tuning"] = results
        return results
    
    # ==================== SIMILARITY METRICS ====================
    
    def compare_similarity_metrics(self, query: str) -> Dict:
        """
        Compare different similarity metrics for the same query
        Cosine, Euclidean, Inner Product
        """
        print(f"\nComparing similarity metrics for: {query}")
        
        results = self.kb.search(query, n_results=5)
        
        comparison = {
            "query": query,
            "metrics": {}
        }
        
        # Cosine similarity (already returned)
        comparison["metrics"]["cosine"] = {
            "results": results["results"],
            "description": "Measures angle between vectors (0-1 range)"
        }
        
        # Note: For other metrics, would need ChromaDB configuration changes
        # This demonstrates the framework for comparison
        
        return comparison
    
    # ==================== SEARCH STRATEGY OPTIMIZATION ====================
    
    def optimize_retrieval_strategy(
        self,
        query: str,
        strategies: List[str] = None
    ) -> Dict:
        if strategies is None:
            strategies = ["simple", "multi", "hyde"]

        print(f"\nOptimizing retrieval strategies for: {query}")

        strategy_results = {}

        for strategy in strategies:
            start = time.time()

            if strategy == "simple":
                search_result = self.kb.search(query, n_results=5)
                result = search_result["results"]

            elif strategy == "multi":
                search_result = self.kb.search(query, n_results=10)
                result = search_result["results"][:5]

            elif strategy == "hyde":
                search_result = self.kb.search(query, n_results=5)
                result = search_result["results"]

            else:
                continue

            elapsed = (time.time() - start) * 1000

            if result and len(result) > 0:
                avg_similarity = np.mean(
                    [r.get("similarity_score", 0) for r in result]
                )
                num_results = len(result)
            else:
                avg_similarity = 0
                num_results = 0

            strategy_results[strategy] = {
                "response_time_ms": round(elapsed, 2),
                "results_count": num_results,
                "avg_similarity": round(avg_similarity, 4),
                "quality_score": round(
                    avg_similarity * (num_results / 5),
                    4
                )
            }

            print(
                f"  {strategy:10s}: "
                f"{elapsed:7.2f}ms | "
                f"Similarity: {avg_similarity:.4f}"
            )

        return {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "strategies": strategy_results
        }
    # ==================== BENCHMARKING ====================
    
    def benchmark_search_quality(self,
                                test_queries: List[str],
                                relevance_judgments: Dict[str, List[str]]) -> Dict:
        """
        Benchmark search quality using precision, recall, NDCG
        
        Args:
            test_queries: List of test queries
            relevance_judgments: Dict mapping query to list of relevant doc IDs
        """
        print(f"\nBenchmarking search quality on {len(test_queries)} queries")
        
        metrics = {
            "queries_tested": len(test_queries),
            "avg_precision": 0,
            "avg_recall": 0,
            "avg_ndcg": 0,
            "avg_response_time_ms": 0,
            "query_results": []
        }
        
        precisions = []
        recalls = []
        ndcgs = []
        times = []
        
        for query in test_queries:
            start = time.time()
            result = self.kb.search(query, n_results=10)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            # Get relevant documents for this query
            relevant_docs = set(relevance_judgments.get(query, []))
            
            if not relevant_docs:
                continue
            
            # Calculate metrics
            retrieved_ids = [r.get('id', r['title'][:20]) for r in result["results"]]
            relevant_retrieved = len([did for did in retrieved_ids if did in relevant_docs])
            
            precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
            
            # NDCG (Normalized Discounted Cumulative Gain)
            dcg = sum([1 / np.log2(i + 2) for i, did in enumerate(retrieved_ids)
                      if did in relevant_docs])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_docs), len(retrieved_ids)))])
            ndcg = dcg / idcg if idcg > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)
            
            metrics["query_results"].append({
                "query": query,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "ndcg": round(ndcg, 4),
                "response_time_ms": round(elapsed, 2)
            })
            
            print(f"  {query[:40]:40s} | P: {precision:.3f} | R: {recall:.3f} | NDCG: {ndcg:.3f}")
        
        metrics["avg_precision"] = round(np.mean(precisions), 4) if precisions else 0
        metrics["avg_recall"] = round(np.mean(recalls), 4) if recalls else 0
        metrics["avg_ndcg"] = round(np.mean(ndcgs), 4) if ndcgs else 0
        metrics["avg_response_time_ms"] = round(np.mean(times), 2) if times else 0
        
        return metrics
    
    def benchmark_latency(self,
                         test_queries: List[str],
                         num_runs: int = 3) -> Dict:
        """
        Benchmark search latency
        """
        print(f"\nBenchmarking latency ({num_runs} runs per query)")
        
        latencies = {}
        
        for query in test_queries:
            times = []
            
            for run in range(num_runs):
                start = time.time()
                self.kb.search(query, n_results=5)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            latencies[query] = {
                "min_ms": round(min(times), 2),
                "max_ms": round(max(times), 2),
                "avg_ms": round(np.mean(times), 2),
                "std_ms": round(np.std(times), 2),
                "p95_ms": round(np.percentile(times, 95), 2),
                "p99_ms": round(np.percentile(times, 99), 2)
            }
            
            print(f"  {query[:40]:40s}: {latencies[query]['avg_ms']:.2f}ms " +
                  f"(p95: {latencies[query]['p95_ms']:.2f}ms)")
        
        return {
            "benchmark_type": "latency",
            "runs_per_query": num_runs,
            "timestamp": datetime.now().isoformat(),
            "queries": latencies
        }
    
    def benchmark_throughput(self,
                            test_queries: List[str],
                            duration_seconds: int = 10) -> Dict:
        """
        Benchmark search throughput (queries per second)
        """
        print(f"\nBenchmarking throughput for {duration_seconds}s")
        
        throughput_results = {}
        
        for query in test_queries:
            query_count = 0
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                self.kb.search(query, n_results=5)
                query_count += 1
            
            qps = query_count / duration_seconds
            throughput_results[query] = {
                "queries": query_count,
                "duration_seconds": duration_seconds,
                "qps": round(qps, 2)
            }
            
            print(f"  {query[:40]:40s}: {qps:.2f} QPS")
        
        return {
            "benchmark_type": "throughput",
            "timestamp": datetime.now().isoformat(),
            "results": throughput_results
        }
    
    # ==================== OPTIMIZATION RECOMMENDATIONS ====================
    
    def get_optimization_recommendations(self) -> Dict:
        """
        Provide recommendations for optimization based on benchmarks
        """
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "recommendations": [],
            "priority_actions": []
        }
        
        if self.tuning_results.get("parameter_tuning"):
            param_results = self.tuning_results["parameter_tuning"]["parameter_variations"]
            
            # Find optimal n_results
            best_config = None
            best_score = -1
            
            for config, metrics in param_results.items():
                score = metrics["avg_quality_score"]
                if metrics["avg_response_time_ms"] < 100 and score > best_score:
                    best_config = config
                    best_score = score
            
            if best_config:
                recommendations["recommendations"].append({
                    "parameter": "n_results",
                    "recommendation": best_config,
                    "reason": "Optimal balance of quality and latency"
                })
                recommendations["priority_actions"].append(
                    f"Use {best_config} for production searches"
                )
        
        # General recommendations
        recommendations["recommendations"].extend([
            {
                "parameter": "Caching",
                "recommendation": "Enable query result caching for repeated queries",
                "reason": "Improves latency for popular queries"
            },
            {
                "parameter": "Batch Processing",
                "recommendation": "Use batch search for multiple queries",
                "reason": "Reduces overhead per query"
            },
            {
                "parameter": "Indexing",
                "recommendation": "Ensure proper HNSW configuration",
                "reason": "Affects search speed and quality trade-off"
            }
        ])
        
        return recommendations
    
    # ==================== VISUALIZATION ====================
    
    def plot_benchmark_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot benchmark results"""
        if "queries" in results:  # Latency benchmark
            queries = list(results["queries"].keys())
            avg_times = [results["queries"][q]["avg_ms"] for q in queries]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(queries)), avg_times)
            plt.xlabel("Query")
            plt.ylabel("Average Latency (ms)")
            plt.title("Search Latency Benchmark")
            plt.xticks(range(len(queries)), [q[:20] for q in queries], rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300)
            plt.show()
    
    def plot_strategy_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Plot strategy comparison"""
        strategies = results.get("strategies", {})
        
        if not strategies:
            return
        
        strategy_names = list(strategies.keys())
        response_times = [strategies[s]["response_time_ms"] for s in strategy_names]
        similarities = [strategies[s]["avg_similarity"] for s in strategy_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(strategy_names, response_times)
        ax1.set_ylabel("Response Time (ms)")
        ax1.set_title("Strategy Comparison: Response Time")
        
        ax2.bar(strategy_names, similarities)
        ax2.set_ylabel("Average Similarity")
        ax2.set_title("Strategy Comparison: Result Quality")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    # ==================== EXPORT & REPORTING ====================
    
    def export_benchmark_results(self, results: Dict, filename: str) -> str:
        """Export benchmark results to JSON"""
        output_path = Path("./tuning_results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results exported to {output_path}")
        return str(output_path)
    
    def generate_tuning_report(self) -> str:
        """Generate comprehensive tuning report"""
        report = []
        report.append("=" * 80)
        report.append("VECTOR SEARCH TUNING REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        
        if self.tuning_results:
            report.append("\n[TUNING RESULTS SUMMARY]")
            for category, results in self.tuning_results.items():
                report.append(f"\n{category}:")
                # Safely serialize numpy / torch types and other non-serializable objects
                report.append(
                    json.dumps(
                        results,
                        indent=2,
                        default=lambda x: (float(x) if hasattr(x, "item") else str(x)),
                    )
                )
        
        # Add recommendations
        recommendations = self.get_optimization_recommendations()
        report.append("\n[OPTIMIZATION RECOMMENDATIONS]")
        for rec in recommendations["recommendations"]:
            report.append(f"  • {rec['parameter']}: {rec['recommendation']}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = Path("./tuning_results/vector_search_report.txt")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        return report_text


# ==================== USAGE EXAMPLES ====================

def main():
    """Example usage"""
    
    print("=" * 80)
    print("WEEK 4: VECTOR SEARCH TUNING EXAMPLES")
    print("=" * 80)
    
    # Initialize tuner
    kb = DataStructuresKB()
    tuner = VectorSearchTuner(kb)
    
    # Example 1: Parameter Tuning
    print("\n[Example 1] Parameter Tuning")
    print("-" * 80)
    tuning_results = tuner.tune_search_parameters(
        query="data structures and algorithms",
        n_results_values=[1, 3, 5, 10, 20]
    )
    
    # Example 2: Strategy Optimization
    print("\n[Example 2] Retrieval Strategy Optimization")
    print("-" * 80)
    strategy_results = tuner.optimize_retrieval_strategy(
        query="What is a binary search tree?"
    )
    print(f"\nBest Strategy Results:")
    for strategy, metrics in strategy_results["strategies"].items():
        print(f"  {strategy}: {metrics}")
    
    # Example 3: Latency Benchmarking
    print("\n[Example 3] Latency Benchmarking")
    print("-" * 80)
    test_queries = [
        "arrays and lists",
        "tree structures",
        "hash tables and hashing"
    ]
    
    latency_results = tuner.benchmark_latency(test_queries, num_runs=3)
    print("\nLatency Results:")
    for query, metrics in latency_results["queries"].items():
        print(f"  {query}: {metrics['avg_ms']:.2f}ms (p95: {metrics['p95_ms']:.2f}ms)")
    
    # Example 4: Quality Benchmarking
    print("\n[Example 4] Search Quality Benchmarking")
    print("-" * 80)
    relevance_judgments = {
        "arrays and lists": ["array_001", "linked_list_001"],
        "tree structures": ["binary_tree_001"],
        "hash tables and hashing": ["hash_table_001"]
    }
    
    quality_results = tuner.benchmark_search_quality(test_queries, relevance_judgments)
    print(f"\nQuality Metrics:")
    print(f"  Avg Precision: {quality_results['avg_precision']:.4f}")
    print(f"  Avg Recall: {quality_results['avg_recall']:.4f}")
    print(f"  Avg NDCG: {quality_results['avg_ndcg']:.4f}")
    
    # Example 5: Get Recommendations
    print("\n[Example 5] Optimization Recommendations")
    print("-" * 80)
    recommendations = tuner.get_optimization_recommendations()
    print("\nRecommendations:")
    for rec in recommendations["recommendations"][:3]:
        print(f"  • {rec['parameter']}: {rec['recommendation']}")
    
    # Example 6: Export Results
    print("\n[Example 6] Export Results")
    print("-" * 80)
    tuner.export_benchmark_results(quality_results, "quality_benchmark.json")
    tuner.export_benchmark_results(latency_results, "latency_benchmark.json")
    
    # Generate Report
    print("\n[Generating Report]")
    print("-" * 80)
    tuner.generate_tuning_report()


if __name__ == "__main__":
    main()
