

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


# def merge_retrieval_results(baseline_results: Dict, embedding_results: Dict) -> Dict:
#     """Merge and deduplicate results from baseline and embeddings."""
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
    
#     # Process embedding results
#     if embedding_results:
#         for node in embedding_results.get("similar_nodes", []):
#             node_id = node.get("id")
#             if node_id:
#                 if node_id in all_nodes:
#                     all_nodes[node_id]["source"] = "both"
#                     all_nodes[node_id]["similarity_score"] = node.get("similarity_score")
#                 else:
#                     node_copy = node.copy()
#                     node_copy["source"] = "embeddings"
#                     all_nodes[node_id] = node_copy
        
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
#     """Format combined retrieval results into human-readable context."""
#     blocks = []
    
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
# Description: {node.get('description')}{source_info}""")
    
#     if relationships:
#         blocks.append("\n--- Relationships ---")
#         for rel in relationships:
#             blocks.append(f"{rel.get('start')} --[{rel.get('type')}]--> {rel.get('end')}")
    
#     if reviews:
#         blocks.append("\n--- Reviews ---")
#         for review in reviews:
#             blocks.append(f"Review for Hotel {review.get('hotel_id')}: \"{review.get('text')}\" (Score: {review.get('score')}/5)")
    
#     return "\n\n".join(blocks)



from typing import Dict, List

def merge_retrieval_results(results: Dict) -> Dict:
    """
    Merge and deduplicate results from baseline, minilm, and mpnet
    """
    all_nodes = {}
    all_reviews = {}

    for source_name, source_data in results.items():
        if not source_data:
            continue

        # -------- Nodes --------
        for node in source_data.get("nodes", []):
            hotel_id = node.get("hotel_id")
            if not hotel_id:
                continue

            if hotel_id not in all_nodes:
                node_copy = node.copy()
                node_copy["id"] = hotel_id
                node_copy["source"] = source_name
                node_copy["reviews"] = []  # ADD THIS: Initialize reviews list
                all_nodes[hotel_id] = node_copy
            else:
                existing = all_nodes[hotel_id]
                existing["source"] = "both"

                if "similarity_score" in node:
                    best = max(
                        existing.get("similarity_score", 0),
                        node.get("similarity_score", 0)
                    )
                    existing["similarity_score"] = best

        # -------- Reviews --------
        # FIX: Associate reviews with their hotels
        hotel_nodes = source_data.get("nodes", [])
        if hotel_nodes:
            # Assuming reviews in this source belong to the hotels in this source
            hotel_id = hotel_nodes[0].get("hotel_id")  # Get the hotel from this source
            
            for review in source_data.get("reviews", []):
                review_id = review.get("review_id")
                if review_id and review_id not in all_reviews:
                    review_copy = review.copy()
                    review_copy["hotel_id"] = hotel_id  # ADD hotel_id to review
                    all_reviews[review_id] = review_copy
                    
                    # Also add to the node's review list
                    if hotel_id in all_nodes:
                        all_nodes[hotel_id]["reviews"].append(review_copy)

    return {
        "nodes": list(all_nodes.values()),
        "reviews": list(all_reviews.values()),
        "total_sources": {
            "baseline_only": sum(
                1 for n in all_nodes.values() if n["source"] == "baseline"
            ),
            "embeddings_only": sum(
                1 for n in all_nodes.values()
                if n["source"] in {"minilm", "mpnet"}
            ),
            "both": sum(
                1 for n in all_nodes.values() if n["source"] == "both"
            ),
        }
    }

# def merge_retrieval_results(results: Dict) -> Dict:
#     """
#     Merge and deduplicate results from baseline, minilm, and mpnet
#     """
#     all_nodes = {}
#     all_reviews = {}

#     for source_name, source_data in results.items():
#         if not source_data:
#             continue

#         # -------- Nodes --------
#         for node in source_data.get("nodes", []):
#             hotel_id = node.get("hotel_id")
#             if not hotel_id:
#                 continue

#             if hotel_id not in all_nodes:
#                 node_copy = node.copy()
#                 node_copy["id"] = hotel_id
#                 node_copy["source"] = source_name
#                 all_nodes[hotel_id] = node_copy
#             else:
#                 # Seen before → mark as multiple sources
#                 existing = all_nodes[hotel_id]
#                 existing["source"] = "both"

#                 # Keep best similarity score if exists
#                 if "similarity_score" in node:
#                     best = max(
#                         existing.get("similarity_score", 0),
#                         node.get("similarity_score", 0)
#                     )
#                     existing["similarity_score"] = best

#         # -------- Reviews --------
#         for review in source_data.get("reviews", []):
#             review_id = review.get("review_id")
#             if review_id:
#                 all_reviews[review_id] = review

#     return {
#         "nodes": list(all_nodes.values()),
#         "reviews": list(all_reviews.values()),
#         "total_sources": {
#             "baseline_only": sum(
#                 1 for n in all_nodes.values() if n["source"] == "baseline"
#             ),
#             "embeddings_only": sum(
#                 1 for n in all_nodes.values()
#                 if n["source"] in {"minilm", "mpnet"}
#             ),
#             "both": sum(
#                 1 for n in all_nodes.values() if n["source"] == "both"
#             ),
#         }
#     }

def format_context(nodes: List[Dict], reviews: List[Dict]) -> str:
    blocks = []

    for node in nodes:
        source = node.get("source", "unknown")
        similarity = node.get("similarity_score")

        source_info = f"[Source: {source}"
        if similarity is not None:
            source_info += f", similarity: {similarity:.2f}"
        source_info += "]"

        hotel_block = f"""Hotel ID: {node.get('id')}
Name: {node.get('name')}
City: {node.get('city')}, {node.get('country')}
Stars: {node.get('star_rating')}
{source_info}"""

        # Add reviews for THIS hotel
        hotel_reviews = node.get("reviews", [])
        if hotel_reviews:
            hotel_block += "\n\nReviews for this hotel:"
            for r in hotel_reviews:
                hotel_block += f"""
  - Review ID: {r.get('review_id')}
    Text: "{r.get('review_text')}"
    Overall Score: {r.get('score_overall')}/10
    Date: {r.get('review_date')}"""
        
        blocks.append(hotel_block)

    return "\n\n".join(blocks)


# def format_context(nodes: List[Dict], reviews: List[Dict]) -> str:
#     blocks = []

#     for node in nodes:
#         source = node.get("source", "unknown")
#         similarity = node.get("similarity_score")

#         source_info = f"[Source: {source}"
#         if similarity is not None:
#             source_info += f", similarity: {similarity:.2f}"
#         source_info += "]"

#         blocks.append(
#             f"""Hotel ID: {node.get('id')}
# Name: {node.get('name')}
# City: {node.get('city')}, {node.get('country')}
# Stars: {node.get('star_rating')}
# {source_info}"""
#         )

#     if reviews:
#         blocks.append("\n--- Reviews ---")
#         for r in reviews:
#             blocks.append(
#                 f"""Review ID: {r.get('review_id')}
# Text: "{r.get('review_text')}"
# Overall Score: {r.get('score_overall')}/10
# Date: {r.get('review_date')}
# """
#             )

#     return "\n\n".join(blocks)


# def format_context(nodes: List[Dict], reviews: List[Dict]) -> str:
#     blocks = []

#     for node in nodes:
#         source = node.get("source", "unknown")
#         similarity = node.get("similarity_score")

#         source_info = f"[Source: {source}"
#         if similarity is not None:
#             source_info += f", similarity: {similarity:.2f}"
#         source_info += "]"

#         blocks.append(
#             f"""Hotel ID: {node.get('id')}
# Name: {node.get('name')}
# City: {node.get('city')}, {node.get('country')}
# Stars: {node.get('star_rating')}
# {source_info}"""
#         )

#     if reviews:
#         blocks.append("\n--- Reviews ---")
#         for r in reviews:
#             blocks.append(
#                 f"""Review ID: {r.get('review_id')}
# Text: "{r.get('review_text')}"
# Overall Score: {r.get('score_overall')}/10
# Date: {r.get('review_date')}
# """
#             )

#     return "\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    """Final optimized prompt - balanced and clear."""
    return f"""You are a helpful hotel information assistant.

HOTEL DATA:
{context}

IMPORTANT INSTRUCTIONS:
1. Answer in natural, conversational language (NOT as raw data or bullet points)
2. Use ONLY the information provided above - do not invent details
3. Do not mention amenities, landmarks, or nearby places unless stated in the data
4. Include all review scores with numbers (e.g., "8.9/10, 9.1/10, 9.0/10")
5. If review text seems unclear or unusual, briefly acknowledge it
6. Write 70-85 words in paragraph format

USER QUESTION: {query}

YOUR ANSWER (write naturally, like speaking to a traveler):"""
# def build_prompt(query: str, context: str) -> str:
#     """Universal prompt optimized for all models."""
#     return f"""You are a professional hotel information assistant providing accurate, helpful responses to travelers.

# HOTEL DATA AVAILABLE:
# {context}

# YOUR TASK:
# Answer the user's question using ONLY the information provided above.

# MANDATORY RULES:
# 1. Base your answer exclusively on the data above - do not invent, assume, or infer any details
# 2. DO NOT mention: amenities (pools, gyms, spas), nearby landmarks, transportation, or features not explicitly stated
# 3. DO NOT make generalizations like "guests loved" or "travelers praised" unless directly quoting a review
# 4. DO NOT interpret unclear review text as hotel features or qualities
# 5. If review text seems unusual or unclear, you may note this briefly but do not try to interpret it
# 6. DO NOT invent any details or reviews
# 7. DO NOT hallucinate

# WHAT TO INCLUDE:
# - Hotel name, city, country, and star rating
# - ALL review scores mentioned (e.g., "scores of 8.9/10, 9.1/10, and 9.0/10")
# - Specific review dates if relevant
# - Direct quotes from reviews and mention if they are meaningless
# - If multiple hotels match, provide information on each and compare between them.


# RESPONSE FORMAT:
# - Write 70-90 words
# - Use natural, conversational language (not bullet points or raw data)
# - Be specific with numbers and dates
# - Acknowledge limitations if data is unclear or insufficient

# GOOD EXAMPLE:
# "The Ocean View Resort is a 4-star hotel in Miami, Florida. Recent reviews from December 2024 show overall scores of 9.2/10, 8.8/10, and 9.5/10. Reviewers consistently rated cleanliness at 9.4/10 and location at 9.6/10. One guest noted 'excellent beachfront access and spacious rooms.' The hotel maintains high scores across all categories based on the available reviews."

# BAD EXAMPLES:
# ❌ "The hotel has stunning views and friendly staff" (inventing details)
# ❌ "Located near the city center and major attractions" (assuming location details)
# ❌ "Guests praised the amenities and service" (vague generalization)
# ❌ "Review ID: 12345, Score: 9.2" (raw data format)

# USER QUESTION: {query}

# YOUR RESPONSE (70-90 words, factual and helpful):"""

# def build_prompt(query: str, context: str) -> str:
#     """Build structured prompt with clear instructions and examples."""
#     return f"""You are a knowledgeable hotel booking assistant helping travelers find their perfect accommodation.

# CONTEXT - Hotel Information Retrieved:
# {context}

# INSTRUCTIONS:
# 1. Answer the user's question naturally and conversationally
# 2. Use ONLY information from the context above - never invent details
# 3. If a detail is not mentioned, say "not specified" or skip it
# 4. NEVER invent details like amenities, nearby landmarks, or features
# 5. Quote review text EXACTLY as written, even if it seems unusual
# 6. If reviews seem unclear, acknowledge that
# 7. Present information in a helpful, narrative format (NOT as raw data dumps)
# 8. Include relevant details like:
#    - Hotel name, location, and star rating
#    - Overall ratings and specific scores (cleanliness, comfort, location, etc.)
#    - Key highlights from recent reviews
#    - Any comparisons if multiple hotels are mentioned
# 9. If information is missing from context, acknowledge it naturally
# 10. Keep your response between 50-150 words
# 11. Be specific with numbers (e.g., "8.9/10" not just "high rating")
# 12. Do not hallucinate

# EXAMPLE GOOD RESPONSE:
# "The Sunset Plaza is a 4-star hotel in Barcelona, Spain. It has an excellent overall rating of 9.2/10, with guests particularly praising its cleanliness (9.5/10) and prime location (9.4/10). Recent reviews highlight the 'stunning rooftop views' and 'attentive staff.' However, some guests noted the rooms could be larger."

# EXAMPLE BAD RESPONSE:
# "Review ID: 12345, Score: 9.2, Date: 2025-12-30" ❌ (Too raw/technical)
# "It's a good hotel." ❌ (Too vague)

# USER QUESTION: {query}

# YOUR RESPONSE:"""

# def build_prompt(query: str, context: str) -> str:
#     """Build structured prompt with persona, context, and task."""
#     return f"""You are a helpful travel assistant specialized in hotel recommendations.

# Context (Retrieved from Knowledge Graph):
# {context}

# Task: Answer the user's question using ONLY the information provided above. Be concise, accurate, and helpful. If the information needed to answer is not in the context, clearly state that you don't have enough information.

# User Question: {query}

# Answer:"""


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
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not found")

    print("\n🔄 Merging retrieval results...")
    merged_data = merge_retrieval_results(
        retrieval_result["results"]
    )

    print(f"   ✅ Combined {len(merged_data['nodes'])} unique nodes")
    print(f"   📊 Sources: {merged_data['total_sources']}")

    context = format_context(
        merged_data["nodes"],
        merged_data["reviews"]
    )
    print("Context!!!")
    print(context)


    query = retrieval_result["query"]
    prompt = build_prompt(query, context)

    client = InferenceClient(token=HF_API_KEY)
    outputs = {}

    for model_name, model_id in FREE_MODELS.items():
        print(f"\n🔄 Querying {model_name}...")
        try:
            start = time.time()

            response = client.chat_completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )

            answer = response.choices[0].message.content.strip()
            elapsed = time.time() - start

            metrics = evaluate_answer_quality(answer, query, context)

            outputs[model_name] = {
                "model_id": model_id,
                "answer": answer,
                "response_time": round(elapsed, 2),
                "quality_metrics": metrics,
                "status": "success"
            }

            print(f"✅ {model_name} done ({elapsed:.2f}s)")

        except Exception as e:
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


def get_ui_response(result: Dict) -> list:
    """
    Returns a simple list of dictionaries with model names and responses.
    
    Returns:
        List of dicts: [{"model_name": str, "response": str}, ...]
    """
    models = []
    
    for model_name, output in result["results"].items():
        if output["status"] == "success":
            models.append({
                "model_name": model_name,
                "response": output["answer"]
            })
        else:
            models.append({
                "model_name": model_name,
                "response": f"Error: {output['error']}"
            })
    
    return models


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

def get_ui_response(result: Dict) -> list:
    """
    Returns a simple list of dictionaries with model names and responses.
    
    Returns:
        List of dicts: [{"model_name": str, "response": str}, ...]
    """
    models = []
    
    for model_name, output in result["results"].items():
        if output["status"] == "success":
            models.append({
                "model_name": model_name,
                "response": output["answer"]
            })
        else:
            models.append({
                "model_name": model_name,
                "response": f"Error: {output['error']}"
            })
    
    return models
if __name__ == "__main__":
    # Test data
    from example_retrieval_result import example_retrieval_result
    
    print("Starting Task 3: LLM Layer")
    print(f"HuggingFace API Key: {'✅ Found' if HF_API_KEY else '❌ Missing'}")
    
    if not HF_API_KEY:
        print("\n⚠️  ERROR: Set HF_API_KEY in .env file")
        exit(1)
    
    result = generate_answers_with_all_models(example_retrieval_result)
    # print(result)
    # print_detailed_comparison(result)

    models_data = get_ui_response(result)
    print(models_data)
    # ui_result=get_ui_response(result)
    # print(ui_result)
    

    

    # print_detailed_comparison(result)
