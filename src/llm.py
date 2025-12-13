# # # llm_robust.py - More robust version with fallback models

# # import os
# # from typing import Dict, List, Optional
# # from dotenv import load_dotenv
# # from huggingface_hub import InferenceClient
# # import time

# # load_dotenv()

# # HF_API_KEY = os.getenv("HF_API_KEY")

# # # Multiple tiers of models to try
# # MODEL_CANDIDATES = {
# #     "Llama-3.2-3B": ["meta-llama/Llama-3.2-3B-Instruct"],
# #     "Llama-3.2-1B": ["meta-llama/Llama-3.2-1B-Instruct"],
# #     "Qwen-2.5-7B": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"],
# #     "Phi-3.5-Mini": ["microsoft/Phi-3.5-mini-instruct"],
# #     "Mistral-7B": ["mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.2"],
# #     "Gemma-2-9B": ["google/gemma-2-9b-it", "google/gemma-2-2b-it"],
# # }


# # def format_context(nodes: List[Dict], extra_context: List[Dict]) -> str:
# #     """Format hotel and review information into readable context."""
# #     blocks = []

# #     for h in nodes:
# #         blocks.append(
# #             f"""Hotel ID: {h['id']}
# # Name: {h['name']}
# # City: {h['city']}, {h['country']}
# # Rating: {h['rating']}/5
# # Amenities: {", ".join(h['amenities'])}
# # Description: {h['description']}"""
# #         )

# #     for r in extra_context:
# #         blocks.append(
# #             f"""Review for {r['hotel_id']}: "{r['text']}" (Score: {r['score']}/5)"""
# #         )

# #     return "\n\n".join(blocks)


# # def build_prompt(query: str, context: str) -> str:
# #     """Build a structured prompt with persona, context, and task."""
# #     return f"""You are a helpful travel assistant specialized in hotel recommendations.

# # Context (Retrieved Information):
# # {context}

# # Task: Answer the user's question using ONLY the information provided above. Be concise and helpful. If you cannot answer based on the context, say so clearly.

# # User Question: {query}

# # Answer:"""


# # def try_model(client: InferenceClient, model_id: str, prompt: str, max_retries: int = 2) -> Optional[str]:
# #     """Try to get a response from a model with retries."""
# #     messages = [{"role": "user", "content": prompt}]
    
# #     for attempt in range(max_retries):
# #         try:
# #             response = client.chat_completion(
# #                 messages=messages,
# #                 model=model_id,
# #                 max_tokens=300,
# #                 temperature=0.1,
# #             )
# #             return response.choices[0].message.content.strip()
# #         except Exception as e:
# #             if attempt < max_retries - 1:
# #                 time.sleep(2)  # Wait before retry
# #             else:
# #                 raise e


# # def generate_answers_with_all_models(retrieval_result: Dict, num_models: int = 3) -> Dict:
# #     """Generate answers from multiple LLM models and compare results."""
    
# #     if not HF_API_KEY:
# #         raise ValueError("HF_API_KEY not found in environment variables")
    
# #     context = format_context(
# #         retrieval_result["nodes"],
# #         retrieval_result.get("extra_context", [])
# #     )

# #     prompt = build_prompt(retrieval_result["query"], context)
    
# #     client = InferenceClient(token=HF_API_KEY)
# #     outputs = {}
# #     successful_models = 0

# #     for model_name, model_candidates in MODEL_CANDIDATES.items():
# #         if successful_models >= num_models:
# #             break
            
# #         print(f"\n🔄 Trying {model_name}...")
        
# #         for model_id in model_candidates:
# #             try:
# #                 print(f"   Testing: {model_id}")
# #                 start_time = time.time()
                
# #                 answer = try_model(client, model_id, prompt)
                
# #                 end_time = time.time()
# #                 response_time = end_time - start_time
                
# #                 outputs[model_name] = {
# #                     "model_id": model_id,
# #                     "answer": answer,
# #                     "response_time": round(response_time, 2),
# #                     "tokens_estimate": len(answer.split()),
# #                     "status": "success"
# #                 }
                
# #                 print(f"✅ {model_name} succeeded ({response_time:.2f}s)")
# #                 successful_models += 1
# #                 break  # Success, move to next model
                
# #             except Exception as e:
# #                 error_msg = str(e)
# #                 print(f"   ⚠️  Failed: {error_msg[:100]}")
                
# #                 # If this was the last candidate, record the failure
# #                 if model_id == model_candidates[-1]:
# #                     outputs[model_name] = {
# #                         "model_id": model_candidates[0],
# #                         "error": error_msg,
# #                         "status": "failed"
# #                     }
        
# #         time.sleep(1)  # Rate limiting

# #     return {
# #         "query": retrieval_result["query"],
# #         "retrieval_method": retrieval_result["retrieval_method"],
# #         "results": outputs
# #     }


# # def evaluate_answer_quality(answer: str, query: str) -> Dict:
# #     """Simple heuristic evaluation of answer quality."""
# #     words = answer.split()
    
# #     # Check if answer mentions key terms from query
# #     query_terms = set(query.lower().split())
# #     answer_terms = set(answer.lower().split())
# #     relevance_score = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0
    
# #     return {
# #         "length": len(words),
# #         "completeness": "complete" if len(words) > 20 else "brief",
# #         "relevance": round(relevance_score, 2),
# #         "mentions_hotels": "hotel" in answer.lower() or "cairo" in answer.lower()
# #     }


# # def print_comparison_report(result: Dict):
# #     """Print a comprehensive comparison report."""
    
# #     print("\n" + "="*80)
# #     print("LLM COMPARISON REPORT")
# #     print("="*80)
    
# #     print(f"\n📝 USER QUERY: {result['query']}")
# #     print(f"🔍 RETRIEVAL METHOD: {result['retrieval_method']}")
    
# #     print("\n" + "="*80)
# #     print("MODEL RESPONSES")
# #     print("="*80)
    
# #     for model_name, output in result["results"].items():
# #         print(f"\n{'─'*80}")
# #         print(f"🤖 MODEL: {model_name}")
# #         print(f"   ID: {output['model_id']}")
        
# #         if output["status"] == "success":
# #             print(f"   ⏱️  Response Time: {output['response_time']}s")
# #             print(f"   📊 Token Estimate: ~{output['tokens_estimate']} words")
            
# #             # Quality evaluation
# #             quality = evaluate_answer_quality(output["answer"], result["query"])
# #             print(f"   📈 Quality Metrics:")
# #             print(f"      - Length: {quality['length']} words ({quality['completeness']})")
# #             print(f"      - Relevance: {quality['relevance']}")
# #             print(f"      - Mentions Key Terms: {'✅' if quality['mentions_hotels'] else '❌'}")
            
# #             print(f"\n📄 ANSWER:")
# #             print(output["answer"])
# #         else:
# #             print(f"   ❌ STATUS: Failed")
# #             print(f"   ERROR: {output['error'][:200]}...")
        
# #         print(f"{'─'*80}")
    
# #     # Quantitative Summary
# #     print("\n" + "="*80)
# #     print("QUANTITATIVE COMPARISON")
# #     print("="*80)
    
# #     successful = [k for k, v in result["results"].items() if v["status"] == "success"]
# #     failed = [k for k, v in result["results"].items() if v["status"] == "failed"]
    
# #     print(f"\n✅ Successful Models: {len(successful)}/{len(result['results'])}")
# #     if successful:
# #         print(f"   {', '.join(successful)}")
    
# #     if failed:
# #         print(f"\n❌ Failed Models: {len(failed)}/{len(result['results'])}")
# #         print(f"   {', '.join(failed)}")
    
# #     if len(successful) >= 2:
# #         print("\n" + "="*80)
# #         print("QUALITATIVE COMPARISON")
# #         print("="*80)
        
# #         if successful:
# #             # Response time comparison
# #             times = {m: result["results"][m]["response_time"] for m in successful}
# #             fastest = min(times, key=times.get)
# #             slowest = max(times, key=times.get)
            
# #             print(f"\n⚡ Speed:")
# #             print(f"   Fastest: {fastest} ({times[fastest]}s)")
# #             print(f"   Slowest: {slowest} ({times[slowest]}s)")
            
# #             # Length comparison
# #             lengths = {m: result["results"][m]["tokens_estimate"] for m in successful}
# #             most_detailed = max(lengths, key=lengths.get)
# #             most_concise = min(lengths, key=lengths.get)
            
# #             print(f"\n📝 Response Length:")
# #             print(f"   Most Detailed: {most_detailed} (~{lengths[most_detailed]} words)")
# #             print(f"   Most Concise: {most_concise} (~{lengths[most_concise]} words)")
            
# #             print(f"\n💡 QUALITATIVE IMPRESSIONS:")
# #             print(f"   - All models successfully grounded responses in provided context")
# #             print(f"   - {fastest} provides fastest responses, good for production")
# #             print(f"   - {most_detailed} provides most comprehensive answers")
# #             print(f"   - Consider cost/quality tradeoff based on your use case")


# # if __name__ == "__main__":
# #     from dummy_input import dummy_retrieval_result
    
# #     print("Starting LLM comparison test with robust fallback...")
# #     print(f"Using HuggingFace API Key: {'✅ Found' if HF_API_KEY else '❌ Missing'}")
    
# #     if not HF_API_KEY:
# #         print("\n⚠️  ERROR: HF_API_KEY not found!")
# #         print("Please set HF_API_KEY in your .env file")
# #         exit(1)
    
# #     result = generate_answers_with_all_models(dummy_retrieval_result, num_models=3)
# #     print_comparison_report(result)


# # task3_llm_complete.py - Complete Task 3 Implementation

# import os
# from typing import Dict, List, Optional, Tuple
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient
# import time
# import json

# load_dotenv()

# HF_API_KEY = os.getenv("HF_API_KEY")

# # Three LLM models for comparison
# FREE_MODELS = {
#     "Llama-3.2-1B": ["meta-llama/Llama-3.2-1B-Instruct"],
#     "Llama-3.2-3B": ["meta-llama/Llama-3.2-3B-Instruct"],
#     # "Gemma-2-2B": "google/gemma-2-2b-it"
#     "Qwen-2.5-7B": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"],

# }


# def merge_retrieval_results(baseline_results: Dict, embedding_results: Dict) -> Dict:
#     """
#     Merge and deduplicate results from baseline (Cypher) and embeddings.
#     Removes duplicates and combines information from both sources.
#     """
#     all_nodes = {}
#     all_relationships = []
#     all_reviews = []
    
#     # Process baseline results
#     if baseline_results:
#         for node in baseline_results.get("nodes", []):
#             node_id = node.get("id")
#             if node_id:
#                 all_nodes[node_id] = node.copy()
#                 all_nodes[node_id]["source"] = "baseline"
        
#         all_relationships.extend(baseline_results.get("relationships", []))
#         all_reviews.extend(baseline_results.get("reviews", []))
    
#     # Process embedding results (avoid duplicates)
#     if embedding_results:
#         for node in embedding_results.get("similar_nodes", []):
#             node_id = node.get("id")
#             if node_id:
#                 if node_id in all_nodes:
#                     # Node found in both - mark as high confidence
#                     all_nodes[node_id]["source"] = "both"
#                     all_nodes[node_id]["similarity_score"] = node.get("similarity_score")
#                 else:
#                     # New node from embeddings
#                     node_copy = node.copy()
#                     node_copy["source"] = "embeddings"
#                     all_nodes[node_id] = node_copy
        
#         # Add reviews from embedding results
#         all_reviews.extend(embedding_results.get("reviews", []))
    
#     # Deduplicate reviews
#     unique_reviews = {(r.get("hotel_id"), r.get("text")): r for r in all_reviews}
    
#     return {
#         "nodes": list(all_nodes.values()),
#         "relationships": all_relationships,
#         "reviews": list(unique_reviews.values()),
#         "total_sources": {
#             "baseline_only": sum(1 for n in all_nodes.values() if n.get("source") == "baseline"),
#             "embeddings_only": sum(1 for n in all_nodes.values() if n.get("source") == "embeddings"),
#             "both": sum(1 for n in all_nodes.values() if n.get("source") == "both"),
#         }
#     }


# def format_context(nodes: List[Dict], relationships: List[Dict], reviews: List[Dict]) -> str:
#     """
#     Format combined retrieval results into human-readable context.
#     Includes nodes, relationships, and reviews with source attribution.
#     """
#     blocks = []
    
#     # Format hotel nodes
#     for node in nodes:
#         source = node.get("source", "unknown")
#         similarity = node.get("similarity_score")
        
#         source_info = f" [Retrieved via: {source}"
#         if similarity:
#             source_info += f", similarity: {similarity:.2f}"
#         source_info += "]"
        
#         blocks.append(f"""Hotel ID: {node.get('id')}
# Name: {node.get('name')}
# City: {node.get('city')}, {node.get('country')}
# Rating: {node.get('rating')}/5
# Amenities: {', '.join(node.get('amenities', []))}
# Description: {node.get('description')}{source_info}""")
    
#     # Format relationships (if any)
#     if relationships:
#         blocks.append("\n--- Relationships ---")
#         for rel in relationships:
#             blocks.append(f"{rel.get('start')} --[{rel.get('type')}]--> {rel.get('end')}")
    
#     # Format reviews
#     if reviews:
#         blocks.append("\n--- Reviews ---")
#         for review in reviews:
#             blocks.append(f"Review for Hotel {review.get('hotel_id')}: \"{review.get('text')}\" (Score: {review.get('score')}/5)")
    
#     return "\n\n".join(blocks)


# def build_prompt(query: str, context: str, persona: str = "travel assistant") -> str:
#     """
#     Build structured prompt with persona, context, and task.
#     This is the key to good LLM performance.
#     """
#     personas = {
#         "travel assistant": "You are a helpful travel assistant specialized in hotel recommendations.",
#         "fpl expert": "You are an FPL (Fantasy Premier League) expert who helps users build winning teams.",
#         "flight assistant": "You are a flight information assistant helping airline companies gain insights."
#     }
    
#     persona_text = personas.get(persona, personas["travel assistant"])
    
#     return f"""{persona_text}

# Context (Retrieved from Knowledge Graph):
# {context}

# Task: Answer the user's question using ONLY the information provided above. Be concise, accurate, and helpful. If the information needed to answer is not in the context, clearly state that you don't have enough information.

# User Question: {query}

# Answer:"""


# def extract_entities_from_answer(answer: str, context: str) -> Dict:
#     """
#     Extract entities mentioned in the answer for accuracy checking.
#     Simple keyword-based extraction.
#     """
#     # Extract hotel names from context
#     context_hotels = set()
#     for line in context.split('\n'):
#         if line.startswith('Name:'):
#             hotel_name = line.replace('Name:', '').strip()
#             context_hotels.add(hotel_name.lower())
    
#     # Check which hotels are mentioned in answer
#     answer_lower = answer.lower()
#     mentioned_hotels = [h for h in context_hotels if h in answer_lower]
    
#     return {
#         "hotels_in_context": len(context_hotels),
#         "hotels_mentioned": len(mentioned_hotels),
#         "coverage": len(mentioned_hotels) / len(context_hotels) if context_hotels else 0
#     }


# def evaluate_answer_quality(answer: str, query: str, context: str) -> Dict:
#     """
#     Comprehensive evaluation of answer quality.
#     Includes both quantitative and qualitative metrics.
#     """
#     metrics = {}
    
#     # Quantitative metrics
#     metrics["length_words"] = len(answer.split())
#     metrics["length_chars"] = len(answer)
    
#     # Entity coverage
#     entity_metrics = extract_entities_from_answer(answer, context)
#     metrics.update(entity_metrics)
    
#     # Query relevance (simple keyword overlap)
#     query_terms = set(query.lower().split())
#     answer_terms = set(answer.lower().split())
#     overlap = query_terms & answer_terms
#     metrics["query_relevance"] = len(overlap) / len(query_terms) if query_terms else 0
    
#     # Hallucination detection (check if answer mentions things not in context)
#     # Simple check: does answer contain unknown hotel names?
#     metrics["likely_grounded"] = all(term in context.lower() for term in answer.split() 
#                                      if term.lower() in ['hotel', 'cairo', 'pool', 'rating'])
    
#     # Completeness indicators
#     metrics["mentions_ratings"] = "rating" in answer.lower() or "/5" in answer
#     metrics["mentions_amenities"] = any(word in answer.lower() for word in ['pool', 'wifi', 'breakfast', 'amenities'])
#     metrics["provides_comparison"] = any(word in answer.lower() for word in ['both', 'compare', 'better', 'higher'])
    
#     # Overall quality score (0-1)
#     quality_factors = [
#         metrics["query_relevance"],
#         metrics["coverage"],
#         1.0 if metrics["mentions_ratings"] else 0.0,
#         1.0 if metrics["mentions_amenities"] else 0.0,
#         1.0 if metrics["likely_grounded"] else 0.0
#     ]
#     metrics["overall_quality"] = sum(quality_factors) / len(quality_factors)
    
#     return metrics


# def generate_answers_with_all_models(retrieval_result: Dict, persona: str = "travel assistant") -> Dict:
#     """
#     Main function: Generate and compare answers from all LLM models.
#     Accepts input from Task 2 (baseline + embeddings).
#     """
    
#     if not HF_API_KEY:
#         raise ValueError("HF_API_KEY not found. Please set it in your .env file")
    
#     # Step 1: Merge baseline and embedding results
#     print("\n🔄 Merging retrieval results...")
#     merged_data = merge_retrieval_results(
#         retrieval_result.get("baseline_results", {}),
#         retrieval_result.get("embedding_results", {})
#     )
    
#     print(f"   ✅ Combined {len(merged_data['nodes'])} unique nodes")
#     print(f"   📊 Sources: {merged_data['total_sources']}")
    
#     # Step 2: Format context
#     context = format_context(
#         merged_data["nodes"],
#         merged_data["relationships"],
#         merged_data["reviews"]
#     )
    
#     # Step 3: Build prompt
#     query = retrieval_result["query"]
#     prompt = build_prompt(query, context, persona)
    
#     # Step 4: Query all models
#     client = InferenceClient(token=HF_API_KEY)
#     outputs = {}
    
#     for model_name, model_id in FREE_MODELS.items():
#         print(f"\n🔄 Querying {model_name}...")
        
#         try:
#             start_time = time.time()
            
#             messages = [{"role": "user", "content": prompt}]
#             response = client.chat_completion(
#                 messages=messages,
#                 model=model_id,
#                 max_tokens=300,
#                 temperature=0.1,
#             )
            
#             end_time = time.time()
#             response_time = end_time - start_time
            
#             answer = response.choices[0].message.content.strip()
            
#             # Evaluate answer quality
#             quality_metrics = evaluate_answer_quality(answer, query, context)
            
#             outputs[model_name] = {
#                 "model_id": model_id,
#                 "answer": answer,
#                 "response_time": round(response_time, 2),
#                 "quality_metrics": quality_metrics,
#                 "status": "success"
#             }
            
#             print(f"✅ {model_name} completed ({response_time:.2f}s, quality: {quality_metrics['overall_quality']:.2f})")
            
#         except Exception as e:
#             print(f"❌ {model_name} failed: {str(e)[:100]}")
#             outputs[model_name] = {
#                 "model_id": model_id,
#                 "error": str(e),
#                 "status": "failed"
#             }
        
#         time.sleep(1)  # Rate limiting
    
#     return {
#         "query": query,
#         "retrieval_stats": {
#             "total_nodes": len(merged_data["nodes"]),
#             "total_reviews": len(merged_data["reviews"]),
#             "source_breakdown": merged_data["total_sources"]
#         },
#         "context": context,
#         "results": outputs
#     }


# def print_detailed_comparison(result: Dict):
#     """Print comprehensive comparison report for presentation."""
    
#     print("\n" + "="*80)
#     print("COMPREHENSIVE LLM COMPARISON REPORT")
#     print("="*80)
    
#     print(f"\n📝 USER QUERY: {result['query']}")
#     print(f"\n📊 RETRIEVAL STATISTICS:")
#     stats = result['retrieval_stats']
#     print(f"   Total Nodes Retrieved: {stats['total_nodes']}")
#     print(f"   Total Reviews: {stats['total_reviews']}")
#     print(f"   Source Breakdown:")
#     for source, count in stats['source_breakdown'].items():
#         print(f"      - {source}: {count}")
    
#     print("\n" + "="*80)
#     print("MODEL RESPONSES & ANALYSIS")
#     print("="*80)
    
#     successful = []
    
#     for model_name, output in result["results"].items():
#         print(f"\n{'─'*80}")
#         print(f"🤖 MODEL: {model_name}")
#         print(f"   ID: {output['model_id']}")
        
#         if output["status"] == "success":
#             successful.append(model_name)
#             metrics = output['quality_metrics']
            
#             print(f"\n⏱️  QUANTITATIVE METRICS:")
#             print(f"   Response Time: {output['response_time']}s")
#             print(f"   Answer Length: {metrics['length_words']} words")
#             print(f"   Entity Coverage: {metrics['coverage']:.2%} ({metrics['hotels_mentioned']}/{metrics['hotels_in_context']} hotels)")
#             print(f"   Query Relevance: {metrics['query_relevance']:.2%}")
#             print(f"   Overall Quality Score: {metrics['overall_quality']:.2%}")
            
#             print(f"\n📈 QUALITATIVE INDICATORS:")
#             print(f"   ✅ Grounded in Context: {'Yes' if metrics['likely_grounded'] else 'No'}")
#             print(f"   ✅ Mentions Ratings: {'Yes' if metrics['mentions_ratings'] else 'No'}")
#             print(f"   ✅ Mentions Amenities: {'Yes' if metrics['mentions_amenities'] else 'No'}")
#             print(f"   ✅ Provides Comparison: {'Yes' if metrics['provides_comparison'] else 'No'}")
            
#             print(f"\n📄 ANSWER:")
#             print(output["answer"])
#         else:
#             print(f"   ❌ ERROR: {output['error'][:150]}")
    
#     # Comparative analysis
#     if len(successful) >= 2:
#         print("\n" + "="*80)
#         print("COMPARATIVE ANALYSIS")
#         print("="*80)
        
#         # Speed comparison
#         times = {m: result["results"][m]["response_time"] for m in successful}
#         fastest = min(times, key=times.get)
#         slowest = max(times, key=times.get)
        
#         print(f"\n⚡ SPEED RANKING:")
#         sorted_by_speed = sorted(times.items(), key=lambda x: x[1])
#         for i, (model, time_val) in enumerate(sorted_by_speed, 1):
#             print(f"   {i}. {model}: {time_val}s")
        
#         # Quality comparison
#         qualities = {m: result["results"][m]["quality_metrics"]["overall_quality"] for m in successful}
#         best_quality = max(qualities, key=qualities.get)
#         worst_quality = min(qualities, key=qualities.get)
        
#         print(f"\n🏆 QUALITY RANKING:")
#         sorted_by_quality = sorted(qualities.items(), key=lambda x: x[1], reverse=True)
#         for i, (model, quality) in enumerate(sorted_by_quality, 1):
#             print(f"   {i}. {model}: {quality:.2%}")
        
#         print(f"\n💡 INSIGHTS:")
#         print(f"   - Fastest Model: {fastest} ({times[fastest]}s)")
#         print(f"   - Highest Quality: {best_quality} ({qualities[best_quality]:.2%})")
        
#         # Recommendation
#         print(f"\n🎯 RECOMMENDATION:")
#         if fastest == best_quality:
#             print(f"   Use {fastest} - Best in both speed and quality!")
#         else:
#             print(f"   For Speed-Critical: Use {fastest}")
#             print(f"   For Quality-Critical: Use {best_quality}")
#             print(f"   For Balanced: Consider cost/quality tradeoff")


# def save_results_for_ui(result: Dict, filename: str = "llm_results.json"):
#     """Save results in format ready for Task 4 UI."""
#     output = {
#         "query": result["query"],
#         "retrieval_stats": result["retrieval_stats"],
#         "context": result["context"],
#         "models": []
#     }
    
#     for model_name, data in result["results"].items():
#         if data["status"] == "success":
#             output["models"].append({
#                 "name": model_name,
#                 "model_id": data["model_id"],
#                 "answer": data["answer"],
#                 "response_time": data["response_time"],
#                 "quality_score": data["quality_metrics"]["overall_quality"],
#                 "metrics": data["quality_metrics"]
#             })
    
#     with open(filename, 'w') as f:
#         json.dump(output, f, indent=2)
    
#     print(f"\n💾 Results saved to {filename} for Task 4 UI")


# if __name__ == "__main__":
#     # Example: Test with dummy data matching Task 2 output format
#     from example_retrieval_result import example_retrieval_result
    
#     print("Starting Task 3: LLM Layer")
#     print(f"HuggingFace API Key: {'✅ Found' if HF_API_KEY else '❌ Missing'}")
    
#     if not HF_API_KEY:
#         print("\n⚠️  ERROR: Set HF_API_KEY in .env file")
#         exit(1)
    
#     # Generate answers
#     result = generate_answers_with_all_models(example_retrieval_result)
    
#     # Print detailed comparison
#     print_detailed_comparison(result)
    
#     # Save for UI
#     save_results_for_ui(result)



# llm_quick_fix.py - Simple working version

import os
from typing import Dict, List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import time
import json

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

# Simple model dictionary - model_id is a STRING, not a list
FREE_MODELS = {
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct"
}


def merge_retrieval_results(baseline_results: Dict, embedding_results: Dict) -> Dict:
    """Merge and deduplicate results from baseline and embeddings."""
    all_nodes = {}
    all_relationships = []
    all_reviews = []
    
    # Process baseline results
    if baseline_results:
        for node in baseline_results.get("nodes", []):
            node_id = node.get("id")
            if node_id:
                all_nodes[node_id] = node.copy()
                all_nodes[node_id]["source"] = "baseline"
        
        all_relationships.extend(baseline_results.get("relationships", []))
        all_reviews.extend(baseline_results.get("reviews", []))
    
    # Process embedding results
    if embedding_results:
        for node in embedding_results.get("similar_nodes", []):
            node_id = node.get("id")
            if node_id:
                if node_id in all_nodes:
                    all_nodes[node_id]["source"] = "both"
                    all_nodes[node_id]["similarity_score"] = node.get("similarity_score")
                else:
                    node_copy = node.copy()
                    node_copy["source"] = "embeddings"
                    all_nodes[node_id] = node_copy
        
        all_reviews.extend(embedding_results.get("reviews", []))
    
    # Deduplicate reviews
    unique_reviews = {(r.get("hotel_id"), r.get("text")): r for r in all_reviews}
    
    return {
        "nodes": list(all_nodes.values()),
        "relationships": all_relationships,
        "reviews": list(unique_reviews.values()),
        "total_sources": {
            "baseline_only": sum(1 for n in all_nodes.values() if n.get("source") == "baseline"),
            "embeddings_only": sum(1 for n in all_nodes.values() if n.get("source") == "embeddings"),
            "both": sum(1 for n in all_nodes.values() if n.get("source") == "both"),
        }
    }


def format_context(nodes: List[Dict], relationships: List[Dict], reviews: List[Dict]) -> str:
    """Format combined retrieval results into human-readable context."""
    blocks = []
    
    for node in nodes:
        source = node.get("source", "unknown")
        similarity = node.get("similarity_score")
        
        source_info = f" [Retrieved via: {source}"
        if similarity:
            source_info += f", similarity: {similarity:.2f}"
        source_info += "]"
        
        blocks.append(f"""Hotel ID: {node.get('id')}
Name: {node.get('name')}
City: {node.get('city')}, {node.get('country')}
Rating: {node.get('rating')}/5
Amenities: {', '.join(node.get('amenities', []))}
Description: {node.get('description')}{source_info}""")
    
    if relationships:
        blocks.append("\n--- Relationships ---")
        for rel in relationships:
            blocks.append(f"{rel.get('start')} --[{rel.get('type')}]--> {rel.get('end')}")
    
    if reviews:
        blocks.append("\n--- Reviews ---")
        for review in reviews:
            blocks.append(f"Review for Hotel {review.get('hotel_id')}: \"{review.get('text')}\" (Score: {review.get('score')}/5)")
    
    return "\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    """Build structured prompt with persona, context, and task."""
    return f"""You are a helpful travel assistant specialized in hotel recommendations.

Context (Retrieved from Knowledge Graph):
{context}

Task: Answer the user's question using ONLY the information provided above. Be concise, accurate, and helpful. If the information needed to answer is not in the context, clearly state that you don't have enough information.

User Question: {query}

Answer:"""


def evaluate_answer_quality(answer: str, query: str, context: str) -> Dict:
    """Comprehensive evaluation of answer quality."""
    metrics = {}
    
    # Basic metrics
    metrics["length_words"] = len(answer.split())
    metrics["length_chars"] = len(answer)
    
    # Extract hotel names from context
    context_hotels = set()
    for line in context.split('\n'):
        if line.startswith('Name:'):
            hotel_name = line.replace('Name:', '').strip()
            context_hotels.add(hotel_name.lower())
    
    # Check which hotels are mentioned in answer
    answer_lower = answer.lower()
    mentioned_hotels = [h for h in context_hotels if h in answer_lower]
    
    metrics["hotels_in_context"] = len(context_hotels)
    metrics["hotels_mentioned"] = len(mentioned_hotels)
    metrics["coverage"] = len(mentioned_hotels) / len(context_hotels) if context_hotels else 0
    
    # Query relevance
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())
    overlap = query_terms & answer_terms
    metrics["query_relevance"] = len(overlap) / len(query_terms) if query_terms else 0
    
    # Quality indicators
    metrics["mentions_ratings"] = "rating" in answer.lower() or "/5" in answer
    metrics["mentions_amenities"] = any(word in answer.lower() for word in ['pool', 'wifi', 'breakfast', 'amenities'])
    metrics["provides_comparison"] = any(word in answer.lower() for word in ['both', 'compare', 'better', 'higher'])
    
    # Overall quality score
    quality_factors = [
        metrics["query_relevance"],
        metrics["coverage"],
        1.0 if metrics["mentions_ratings"] else 0.0,
        1.0 if metrics["mentions_amenities"] else 0.0,
    ]
    metrics["overall_quality"] = sum(quality_factors) / len(quality_factors)
    
    return metrics


def generate_answers_with_all_models(retrieval_result: Dict) -> Dict:
    """Generate and compare answers from all LLM models."""
    
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not found. Please set it in your .env file")
    
    print("\n🔄 Merging retrieval results...")
    merged_data = merge_retrieval_results(
        retrieval_result.get("baseline_results", {}),
        retrieval_result.get("embedding_results", {})
    )
    
    print(f"   ✅ Combined {len(merged_data['nodes'])} unique nodes")
    print(f"   📊 Sources: {merged_data['total_sources']}")
    
    context = format_context(
        merged_data["nodes"],
        merged_data["relationships"],
        merged_data["reviews"]
    )
    
    query = retrieval_result["query"]
    prompt = build_prompt(query, context)
    
    client = InferenceClient(token=HF_API_KEY)
    outputs = {}
    
    for model_name, model_id in FREE_MODELS.items():
        print(f"\n🔄 Querying {model_name}...")
        print(f"   Model ID type: {type(model_id)}, value: {model_id}")  # Debug line
        
        try:
            start_time = time.time()
            
            messages = [{"role": "user", "content": prompt}]
            response = client.chat_completion(
                messages=messages,
                model=model_id,  # This MUST be a string
                max_tokens=300,
                temperature=0.1,
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            answer = response.choices[0].message.content.strip()
            quality_metrics = evaluate_answer_quality(answer, query, context)
            
            outputs[model_name] = {
                "model_id": model_id,
                "answer": answer,
                "response_time": round(response_time, 2),
                "quality_metrics": quality_metrics,
                "status": "success"
            }
            
            print(f"✅ {model_name} completed ({response_time:.2f}s, quality: {quality_metrics['overall_quality']:.2f})")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
            outputs[model_name] = {
                "model_id": model_id,
                "error": str(e),
                "status": "failed"
            }
        
        time.sleep(1)
    
    return {
        "query": query,
        "retrieval_stats": {
            "total_nodes": len(merged_data["nodes"]),
            "total_reviews": len(merged_data["reviews"]),
            "source_breakdown": merged_data["total_sources"]
        },
        "context": context,
        "results": outputs
    }


def print_detailed_comparison(result: Dict):
    """Print comprehensive comparison report."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE LLM COMPARISON REPORT")
    print("="*80)
    
    print(f"\n📝 USER QUERY: {result['query']}")
    print(f"\n📊 RETRIEVAL STATISTICS:")
    stats = result['retrieval_stats']
    print(f"   Total Nodes Retrieved: {stats['total_nodes']}")
    print(f"   Total Reviews: {stats['total_reviews']}")
    print(f"   Source Breakdown:")
    for source, count in stats['source_breakdown'].items():
        print(f"      - {source}: {count}")
    
    print("\n" + "="*80)
    print("MODEL RESPONSES & ANALYSIS")
    print("="*80)
    
    successful = []
    
    for model_name, output in result["results"].items():
        print(f"\n{'─'*80}")
        print(f"🤖 MODEL: {model_name}")
        print(f"   ID: {output['model_id']}")
        
        if output["status"] == "success":
            successful.append(model_name)
            metrics = output['quality_metrics']
            
            print(f"\n⏱️  QUANTITATIVE METRICS:")
            print(f"   Response Time: {output['response_time']}s")
            print(f"   Answer Length: {metrics['length_words']} words")
            print(f"   Entity Coverage: {metrics['coverage']:.2%} ({metrics['hotels_mentioned']}/{metrics['hotels_in_context']} hotels)")
            print(f"   Query Relevance: {metrics['query_relevance']:.2%}")
            print(f"   Overall Quality Score: {metrics['overall_quality']:.2%}")
            
            print(f"\n📈 QUALITATIVE INDICATORS:")
            print(f"   ✅ Mentions Ratings: {'Yes' if metrics['mentions_ratings'] else 'No'}")
            print(f"   ✅ Mentions Amenities: {'Yes' if metrics['mentions_amenities'] else 'No'}")
            print(f"   ✅ Provides Comparison: {'Yes' if metrics['provides_comparison'] else 'No'}")
            
            print(f"\n📄 ANSWER:")
            print(output["answer"])
        else:
            print(f"   ❌ ERROR: {output['error'][:200]}")
    
    # Comparative analysis
    if len(successful) >= 2:
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        times = {m: result["results"][m]["response_time"] for m in successful}
        fastest = min(times, key=times.get)
        
        print(f"\n⚡ SPEED RANKING:")
        sorted_by_speed = sorted(times.items(), key=lambda x: x[1])
        for i, (model, time_val) in enumerate(sorted_by_speed, 1):
            print(f"   {i}. {model}: {time_val}s")
        
        qualities = {m: result["results"][m]["quality_metrics"]["overall_quality"] for m in successful}
        best_quality = max(qualities, key=qualities.get)
        
        print(f"\n🏆 QUALITY RANKING:")
        sorted_by_quality = sorted(qualities.items(), key=lambda x: x[1], reverse=True)
        for i, (model, quality) in enumerate(sorted_by_quality, 1):
            print(f"   {i}. {model}: {quality:.2%}")
        
        print(f"\n🎯 RECOMMENDATION:")
        if fastest == best_quality:
            print(f"   Use {fastest} - Best in both speed and quality!")
        else:
            print(f"   For Speed: Use {fastest}")
            print(f"   For Quality: Use {best_quality}")


if __name__ == "__main__":
    # Test data
    from example_retrieval_result import example_retrieval_result
    
    print("Starting Task 3: LLM Layer")
    print(f"HuggingFace API Key: {'✅ Found' if HF_API_KEY else '❌ Missing'}")
    
    if not HF_API_KEY:
        print("\n⚠️  ERROR: Set HF_API_KEY in .env file")
        exit(1)
    
    result = generate_answers_with_all_models(example_retrieval_result)
    print_detailed_comparison(result)