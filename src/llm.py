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
    "Qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Gemma-2B": "google/gemma-2-2b-it",                  # Safety-focused
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


# def print_detailed_comparison(result: Dict):
#     """Print comprehensive comparison report."""
    
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
#             print(f"   ✅ Mentions Ratings: {'Yes' if metrics['mentions_ratings'] else 'No'}")
#             print(f"   ✅ Mentions Amenities: {'Yes' if metrics['mentions_amenities'] else 'No'}")
#             print(f"   ✅ Provides Comparison: {'Yes' if metrics['provides_comparison'] else 'No'}")
            
#             print(f"\n📄 ANSWER:")
#             print(output["answer"])
#         else:
#             print(f"   ❌ ERROR: {output['error'][:200]}")
    
#     # Comparative analysis
#     if len(successful) >= 2:
#         print("\n" + "="*80)
#         print("COMPARATIVE ANALYSIS")
#         print("="*80)
        
#         times = {m: result["results"][m]["response_time"] for m in successful}
#         fastest = min(times, key=times.get)
        
#         print(f"\n⚡ SPEED RANKING:")
#         sorted_by_speed = sorted(times.items(), key=lambda x: x[1])
#         for i, (model, time_val) in enumerate(sorted_by_speed, 1):
#             print(f"   {i}. {model}: {time_val}s")
        
#         qualities = {m: result["results"][m]["quality_metrics"]["overall_quality"] for m in successful}
#         best_quality = max(qualities, key=qualities.get)
        
#         print(f"\n🏆 QUALITY RANKING:")
#         sorted_by_quality = sorted(qualities.items(), key=lambda x: x[1], reverse=True)
#         for i, (model, quality) in enumerate(sorted_by_quality, 1):
#             print(f"   {i}. {model}: {quality:.2%}")
        
#         print(f"\n🎯 RECOMMENDATION:")
#         if fastest == best_quality:
#             print(f"   Use {fastest} - Best in both speed and quality!")
#         else:
#             print(f"   For Speed: Use {fastest}")
#             print(f"   For Quality: Use {best_quality}")


def evaluate_answer_quality(answer: str, query: str, context: str) -> Dict:
    """
    Comprehensive evaluation with both quantitative and qualitative metrics.
    
    Quantitative: Length, coverage, relevance, token usage
    Qualitative: Correctness, naturalness, helpfulness
    """
    import re
    
    metrics = {}
    
    # ============================================
    # QUANTITATIVE METRICS
    # ============================================
    
    # 1. Length metrics
    metrics["length_words"] = len(answer.split())
    metrics["length_chars"] = len(answer)
    metrics["length_sentences"] = max(1, answer.count('.') + answer.count('!') + answer.count('?'))
    
    # 2. Token usage estimation (approximate: 1 token ≈ 4 chars)
    metrics["estimated_tokens"] = len(answer) // 4
    
    # 3. Entity coverage (how many hotels mentioned)
    context_hotels = set()
    for line in context.split('\n'):
        if line.startswith('Name:'):
            hotel_name = line.replace('Name:', '').strip()
            context_hotels.add(hotel_name.lower())
    
    answer_lower = answer.lower()
    mentioned_hotels = [h for h in context_hotels if h in answer_lower]
    
    metrics["hotels_in_context"] = len(context_hotels)
    metrics["hotels_mentioned"] = len(mentioned_hotels)
    metrics["coverage"] = len(mentioned_hotels) / len(context_hotels) if context_hotels else 0
    
    # 4. Query relevance (keyword overlap)
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())
    overlap = query_terms & answer_terms
    metrics["query_relevance"] = len(overlap) / len(query_terms) if query_terms else 0
    
    # 5. Information density (words per sentence)
    metrics["info_density"] = metrics["length_words"] / metrics["length_sentences"]
    
    # ============================================
    # QUALITATIVE INDICATORS (Automated)
    # ============================================
    
    # 6. Correctness indicators
    metrics["mentions_ratings"] = "rating" in answer.lower() or "/5" in answer
    metrics["mentions_location"] = any(word in answer.lower() for word in ['city', 'location', 'near', 'cairo', 'egypt'])
    metrics["provides_comparison"] = any(word in answer.lower() for word in ['both', 'compare', 'better', 'higher', 'vs', 'whereas'])
    
    # 7. Grounding check
    context_lower = context.lower()
    answer_numbers = re.findall(r'\b\d+\.?\d*\b', answer)
    context_numbers = re.findall(r'\b\d+\.?\d*\b', context)
    
    grounded_numbers = sum(1 for num in answer_numbers if num in context_numbers)
    metrics["grounding_score"] = grounded_numbers / len(answer_numbers) if answer_numbers else 1.0
    
    # 8. Naturalness indicators
    metrics["uses_natural_language"] = not answer.startswith('{') and not answer.startswith('[')
    metrics["has_conclusion"] = any(word in answer.lower() for word in ['therefore', 'overall', 'in summary', 'recommend', 'suggest'])
    metrics["conversational_tone"] = any(word in answer for word in ['I', 'you', 'your', 'would', 'should'])
    
    # 9. Helpfulness indicators
    metrics["actionable"] = any(word in answer.lower() for word in ['recommend', 'suggest', 'consider', 'choose', 'best'])
    metrics["provides_reasoning"] = any(word in answer.lower() for word in ['because', 'since', 'due to', 'as', 'given'])
    metrics["handles_uncertainty"] = any(phrase in answer.lower() for phrase in 
                                          ["don't have", "not available", "cannot determine", "unclear", "not enough"])
    
    # 10. Answer completeness
    if metrics["length_words"] < 10:
        metrics["completeness"] = "too_brief"
    elif metrics["length_words"] > 200:
        metrics["completeness"] = "too_verbose"
    else:
        metrics["completeness"] = "appropriate"
    
    # ============================================
    # OVERALL SCORES
    # ============================================
    
    # Accuracy score
    accuracy_factors = [
        metrics["grounding_score"],
        metrics["coverage"],
        1.0 if metrics["mentions_ratings"] else 0.5,
        1.0 if metrics["mentions_location"] else 0.5,
    ]
    metrics["accuracy_score"] = sum(accuracy_factors) / len(accuracy_factors)
    
    # Relevance score
    relevance_factors = [
        metrics["query_relevance"],
        metrics["coverage"],
        1.0 if metrics["actionable"] else 0.5,
    ]
    metrics["relevance_score"] = sum(relevance_factors) / len(relevance_factors)
    
    # Naturalness score
    naturalness_factors = [
        1.0 if metrics["uses_natural_language"] else 0.0,
        1.0 if metrics["conversational_tone"] else 0.5,
        1.0 if metrics["has_conclusion"] else 0.5,
        1.0 if metrics["completeness"] == "appropriate" else 0.5,
    ]
    metrics["naturalness_score"] = sum(naturalness_factors) / len(naturalness_factors)
    
    # Overall quality score (weighted average)
    metrics["overall_quality"] = (
        metrics["accuracy_score"] * 0.4 +
        metrics["relevance_score"] * 0.3 +
        metrics["naturalness_score"] * 0.3
    )
    
    return metrics

def print_detailed_comparison(result: Dict):
    """Print comprehensive comparison report with all metrics."""
    
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
            
            # ============================================
            # QUANTITATIVE METRICS
            # ============================================
            print(f"\n📊 QUANTITATIVE METRICS:")
            print(f"   Response Time: {output['response_time']}s")
            print(f"   Token Usage (estimated): ~{metrics['estimated_tokens']} tokens")
            print(f"   Answer Length: {metrics['length_words']} words, {metrics['length_sentences']} sentences")
            print(f"   Information Density: {metrics['info_density']:.1f} words/sentence")
            print(f"   Entity Coverage: {metrics['coverage']:.2%} ({metrics['hotels_mentioned']}/{metrics['hotels_in_context']} hotels)")
            print(f"   Query Relevance: {metrics['query_relevance']:.2%}")
            print(f"   Grounding Score: {metrics['grounding_score']:.2%}")
            
            # ============================================
            # QUALITATIVE INDICATORS
            # ============================================
            print(f"\n📈 QUALITATIVE INDICATORS:")
            print(f"   Accuracy Score: {metrics['accuracy_score']:.2%}")
            print(f"   Relevance Score: {metrics['relevance_score']:.2%}")
            print(f"   Naturalness Score: {metrics['naturalness_score']:.2%}")
            print(f"   Overall Quality: {metrics['overall_quality']:.2%}")
            
            print(f"\n✅ CORRECTNESS:")
            print(f"   • Mentions Ratings: {'Yes' if metrics['mentions_ratings'] else 'No'}")
            print(f"   • Mentions Location: {'Yes' if metrics['mentions_location'] else 'No'}")
            print(f"   • Provides Comparison: {'Yes' if metrics['provides_comparison'] else 'No'}")
            print(f"   • Grounded in Context: {metrics['grounding_score']:.0%}")
            
            print(f"\n💬 NATURALNESS:")
            print(f"   • Natural Language: {'Yes' if metrics['uses_natural_language'] else 'No'}")
            print(f"   • Conversational Tone: {'Yes' if metrics['conversational_tone'] else 'No'}")
            print(f"   • Has Conclusion: {'Yes' if metrics['has_conclusion'] else 'No'}")
            print(f"   • Completeness: {metrics['completeness'].replace('_', ' ').title()}")
            
            print(f"\n🎯 HELPFULNESS:")
            print(f"   • Actionable Advice: {'Yes' if metrics['actionable'] else 'No'}")
            print(f"   • Provides Reasoning: {'Yes' if metrics['provides_reasoning'] else 'No'}")
            print(f"   • Handles Uncertainty: {'Yes' if metrics['handles_uncertainty'] else 'No'}")
            
            print(f"\n📄 ANSWER:")
            print(output["answer"])
        else:
            print(f"   ❌ ERROR: {output['error'][:200]}")
    
    # ============================================
    # COMPARATIVE ANALYSIS
    # ============================================
    if len(successful) >= 2:
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        # Speed ranking
        times = {m: result["results"][m]["response_time"] for m in successful}
        fastest = min(times, key=times.get)
        slowest = max(times, key=times.get)
        
        print(f"\n⚡ SPEED RANKING:")
        sorted_by_speed = sorted(times.items(), key=lambda x: x[1])
        for i, (model, time_val) in enumerate(sorted_by_speed, 1):
            print(f"   {i}. {model}: {time_val}s")
        
        # Token usage ranking
        tokens = {m: result["results"][m]["quality_metrics"]["estimated_tokens"] for m in successful}
        most_tokens = max(tokens, key=tokens.get)
        least_tokens = min(tokens, key=tokens.get)
        
        print(f"\n🔢 TOKEN USAGE:")
        sorted_by_tokens = sorted(tokens.items(), key=lambda x: x[1])
        for i, (model, token_count) in enumerate(sorted_by_tokens, 1):
            print(f"   {i}. {model}: ~{token_count} tokens")
        
        # Quality rankings
        print(f"\n🏆 QUALITY RANKINGS:")
        
        # Overall quality
        qualities = {m: result["results"][m]["quality_metrics"]["overall_quality"] for m in successful}
        print(f"\n   Overall Quality:")
        sorted_by_quality = sorted(qualities.items(), key=lambda x: x[1], reverse=True)
        for i, (model, quality) in enumerate(sorted_by_quality, 1):
            print(f"   {i}. {model}: {quality:.2%}")
        best_quality = sorted_by_quality[0][0]
        
        # Accuracy
        accuracies = {m: result["results"][m]["quality_metrics"]["accuracy_score"] for m in successful}
        print(f"\n   Accuracy:")
        sorted_by_accuracy = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
        for i, (model, acc) in enumerate(sorted_by_accuracy, 1):
            print(f"   {i}. {model}: {acc:.2%}")
        most_accurate = sorted_by_accuracy[0][0]
        
        # Relevance
        relevances = {m: result["results"][m]["quality_metrics"]["relevance_score"] for m in successful}
        print(f"\n   Relevance:")
        sorted_by_relevance = sorted(relevances.items(), key=lambda x: x[1], reverse=True)
        for i, (model, rel) in enumerate(sorted_by_relevance, 1):
            print(f"   {i}. {model}: {rel:.2%}")
        most_relevant = sorted_by_relevance[0][0]
        
        # Naturalness
        naturalnesses = {m: result["results"][m]["quality_metrics"]["naturalness_score"] for m in successful}
        print(f"\n   Naturalness:")
        sorted_by_naturalness = sorted(naturalnesses.items(), key=lambda x: x[1], reverse=True)
        for i, (model, nat) in enumerate(sorted_by_naturalness, 1):
            print(f"   {i}. {model}: {nat:.2%}")
        most_natural = sorted_by_naturalness[0][0]
        
        # ============================================
        # COST ANALYSIS (Estimated)
        # ============================================
        print(f"\n💰 ESTIMATED COST ANALYSIS (if using paid API):")
        print(f"   Note: These are FREE on HuggingFace, but here's what it would cost elsewhere:")
        
        # Typical pricing: ~$0.10-0.50 per 1M tokens for small models
        for model in successful:
            token_count = result["results"][model]["quality_metrics"]["estimated_tokens"]
            # Rough estimate: smaller models = cheaper
            if "1B" in model:
                cost_per_1m = 0.10
            elif "3B" in model:
                cost_per_1m = 0.20
            else:  # 7B+
                cost_per_1m = 0.50
            
            estimated_cost = (token_count / 1_000_000) * cost_per_1m
            print(f"   {model}: ${estimated_cost:.6f} per query (~${estimated_cost * 1000:.2f} per 1K queries)")
        
        # ============================================
        # RECOMMENDATIONS
        # ============================================
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"\n   Best Overall: {best_quality}")
        print(f"   Fastest: {fastest} ({times[fastest]:.2f}s)")
        print(f"   Most Accurate: {most_accurate} ({accuracies[most_accurate]:.2%})")
        print(f"   Most Relevant: {most_relevant} ({relevances[most_relevant]:.2%})")
        print(f"   Most Natural: {most_natural} ({naturalnesses[most_natural]:.2%})")
        print(f"   Most Efficient (tokens): {least_tokens} (~{tokens[least_tokens]} tokens)")
        
        print(f"\n   💡 USE CASE RECOMMENDATIONS:")
        print(f"   • Production (balanced): {best_quality}")
        print(f"   • Speed-critical: {fastest}")
        print(f"   • Quality-critical: {most_accurate}")
        print(f"   • Cost-sensitive: {least_tokens}")
        
        # Trade-off analysis
        print(f"\n   ⚖️  TRADE-OFF ANALYSIS:")
        if fastest == best_quality:
            print(f"   ✨ {fastest} is both fastest AND highest quality - IDEAL!")
        else:
            speed_diff = times[slowest] - times[fastest]
            quality_diff = qualities[best_quality] - qualities[fastest]
            print(f"   • {best_quality} is {quality_diff:.1%} better quality but {speed_diff:.2f}s slower")
            print(f"   • {fastest} is {speed_diff:.2f}s faster but {quality_diff:.1%} lower quality")
            
            if quality_diff < 0.10:  # Less than 10% quality difference
                print(f"   ➡️  Recommendation: Use {fastest} (quality difference is minimal)")
            else:
                print(f"   ➡️  Recommendation: Use {best_quality} for quality, {fastest} for speed")




# ## 📋 **Complete Metrics Summary**

# ### **Quantitative Metrics** ✅
# | Metric | What it Measures | How |
# |--------|------------------|-----|
# | **Response Time** | Speed of answer generation | Actual time measurement |
# | **Token Usage** | API cost/resource usage | Character count / 4 |
# | **Answer Length** | Verbosity (words, sentences) | Word/sentence count |
# | **Information Density** | Words per sentence | Words / sentences |
# | **Entity Coverage** | How many hotels mentioned | Hotels mentioned / total |
# | **Query Relevance** | Keyword overlap with query | Jaccard similarity |
# | **Grounding Score** | Factual accuracy | Numbers in answer found in context |

# ### **Qualitative Metrics** ✅
# | Metric | What it Measures | How |
# |--------|------------------|-----|
# | **Accuracy Score** | Correctness of information | Grounding + coverage + facts |
# | **Relevance Score** | How well it answers query | Query relevance + actionability |
# | **Naturalness Score** | Human-like quality | Conversational + completeness |
# | **Overall Quality** | Weighted combination | 40% accuracy + 30% relevance + 30% naturalness |
# | **Correctness Indicators** | Mentions ratings, locations, etc | Keyword detection |
# | **Naturalness Indicators** | Conversational tone, conclusions | Pattern matching |
# | **Helpfulness Indicators** | Actionable advice, reasoning | Keyword detection |

# ---

# ## 🎯 **Sample Output**
# ```
# ================================================================================
# COMPREHENSIVE LLM COMPARISON REPORT
# ================================================================================

# 📝 USER QUERY: Find hotels in Cairo with pools

# 📊 RETRIEVAL STATISTICS:
#    Total Nodes Retrieved: 3
#    Total Reviews: 2
#    Source Breakdown:
#       - baseline_only: 1
#       - embeddings_only: 1
#       - both: 1

# ================================================================================
# MODEL RESPONSES & ANALYSIS
# ================================================================================

# ────────────────────────────────────────────────────────────────────────────────
# 🤖 MODEL: Llama-3.2-3B
#    ID: meta-llama/Llama-3.2-3B-Instruct

# 📊 QUANTITATIVE METRICS:
#    Response Time: 1.23s
#    Token Usage (estimated): ~187 tokens
#    Answer Length: 75 words, 4 sentences
#    Information Density: 18.8 words/sentence
#    Entity Coverage: 66.67% (2/3 hotels)
#    Query Relevance: 75.00%
#    Grounding Score: 100.00%

# 📈 QUALITATIVE INDICATORS:
#    Accuracy Score: 87.50%
#    Relevance Score: 80.56%
#    Naturalness Score: 87.50%
#    Overall Quality: 85.42%

# ✅ CORRECTNESS:
#    • Mentions Ratings: Yes
#    • Mentions Location: Yes
#    • Provides Comparison: Yes
#    • Grounded in Context: 100%

# 💬 NATURALNESS:
#    • Natural Language: Yes
#    • Conversational Tone: Yes
#    • Has Conclusion: Yes
#    • Completeness: Appropriate

# 🎯 HELPFULNESS:
#    • Actionable Advice: Yes
#    • Provides Reasoning: Yes
#    • Handles Uncertainty: No

# 📄 ANSWER:
# Based on the retrieved information, I found two hotels in Cairo with pools:
# Nile Plaza (4.5/5 rating) and Cairo Grand (4.3/5). Nile Plaza has a higher
# rating and is located near the Egyptian Museum. I recommend Nile Plaza for
# its excellent rating and convenient location.

# ================================================================================
# COMPARATIVE ANALYSIS
# ================================================================================

# ⚡ SPEED RANKING:
#    1. Llama-3.2-1B: 0.87s
#    2. Llama-3.2-3B: 1.23s
#    3. Qwen-2.5-7B: 3.45s

# 🔢 TOKEN USAGE:
#    1. Llama-3.2-1B: ~142 tokens
#    2. Llama-3.2-3B: ~187 tokens
#    3. Qwen-2.5-7B: ~215 tokens

# 🏆 QUALITY RANKINGS:

#    Overall Quality:
#    1. Qwen-2.5-7B: 91.23%
#    2. Llama-3.2-3B: 85.42%
#    3. Llama-3.2-1B: 78.34%

#    Accuracy:
#    1. Qwen-2.5-7B: 93.75%
#    2. Llama-3.2-3B: 87.50%
#    3. Llama-3.2-1B: 81.25%

#    Relevance:
#    1. Qwen-2.5-7B: 88.89%
#    2. Llama-3.2-3B: 80.56%
#    3. Llama-3.2-1B: 75.00%

#    Naturalness:
#    1. Llama-3.2-3B: 87.50%
#    2. Qwen-2.5-7B: 87.50%
#    3. Llama-3.2-1B: 75.00%

# # 💰 ESTIMATED COST ANALYSIS (if using paid API):
# #    Note: These are FREE on HuggingFace, but here's what it would cost elsewhere:
# #    Llama-3.2-1B: $0.000014 per query (~$0.01 per 1K queries)
# #    Llama-3.2-3B: $0.000037 per query (~$0.04 per 1K queries)
# #    Qwen-2.5-7B: $0.000108 per query (~$0.11 per 1K queries)

# 🎯 RECOMMENDATIONS:

#    Best Overall: Qwen-2.5-7B
#    Fastest: Llama-3.2-1B (0.87s)
#    Most Accurate: Qwen-2.5-7B (93.75%)
#    Most Relevant: Qwen-2.5-7B (88.89%)
#    Most Natural: Llama-3.2-3B (87.50%)
#    Most Efficient (tokens): Llama-3.2-1B (~142 tokens)

#    💡 USE CASE RECOMMENDATIONS:
#    • Production (balanced): Qwen-2.5-7B
#    • Speed-critical: Llama-3.2-1B
#    • Quality-critical: Qwen-2.5-7B
#    • Cost-sensitive: Llama-3.2-1B

#    ⚖️  TRADE-OFF ANALYSIS:
#    • Qwen-2.5-7B is 5.8% better quality but 2.58s slower
#    • Llama-3.2-1B is 2.58s faster but 5.8% lower quality
#    ➡️  Recommendation: Use Qwen-2.5-7B for quality, Llama-3.2-1B for speed
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