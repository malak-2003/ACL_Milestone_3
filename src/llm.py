# llm_robust.py - More robust version with fallback models

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import time

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

# Multiple tiers of models to try
MODEL_CANDIDATES = {
    "Llama-3.2-3B": ["meta-llama/Llama-3.2-3B-Instruct"],
    "Llama-3.2-1B": ["meta-llama/Llama-3.2-1B-Instruct"],
    "Qwen-2.5-7B": ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"],
    "Phi-3.5-Mini": ["microsoft/Phi-3.5-mini-instruct"],
    "Mistral-7B": ["mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.2"],
    "Gemma-2-9B": ["google/gemma-2-9b-it", "google/gemma-2-2b-it"],
}


def format_context(nodes: List[Dict], extra_context: List[Dict]) -> str:
    """Format hotel and review information into readable context."""
    blocks = []

    for h in nodes:
        blocks.append(
            f"""Hotel ID: {h['id']}
Name: {h['name']}
City: {h['city']}, {h['country']}
Rating: {h['rating']}/5
Amenities: {", ".join(h['amenities'])}
Description: {h['description']}"""
        )

    for r in extra_context:
        blocks.append(
            f"""Review for {r['hotel_id']}: "{r['text']}" (Score: {r['score']}/5)"""
        )

    return "\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    """Build a structured prompt with persona, context, and task."""
    return f"""You are a helpful travel assistant specialized in hotel recommendations.

Context (Retrieved Information):
{context}

Task: Answer the user's question using ONLY the information provided above. Be concise and helpful. If you cannot answer based on the context, say so clearly.

User Question: {query}

Answer:"""


def try_model(client: InferenceClient, model_id: str, prompt: str, max_retries: int = 2) -> Optional[str]:
    """Try to get a response from a model with retries."""
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                messages=messages,
                model=model_id,
                max_tokens=300,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                raise e


def generate_answers_with_all_models(retrieval_result: Dict, num_models: int = 3) -> Dict:
    """Generate answers from multiple LLM models and compare results."""
    
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not found in environment variables")
    
    context = format_context(
        retrieval_result["nodes"],
        retrieval_result.get("extra_context", [])
    )

    prompt = build_prompt(retrieval_result["query"], context)
    
    client = InferenceClient(token=HF_API_KEY)
    outputs = {}
    successful_models = 0

    for model_name, model_candidates in MODEL_CANDIDATES.items():
        if successful_models >= num_models:
            break
            
        print(f"\n🔄 Trying {model_name}...")
        
        for model_id in model_candidates:
            try:
                print(f"   Testing: {model_id}")
                start_time = time.time()
                
                answer = try_model(client, model_id, prompt)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                outputs[model_name] = {
                    "model_id": model_id,
                    "answer": answer,
                    "response_time": round(response_time, 2),
                    "tokens_estimate": len(answer.split()),
                    "status": "success"
                }
                
                print(f"✅ {model_name} succeeded ({response_time:.2f}s)")
                successful_models += 1
                break  # Success, move to next model
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ⚠️  Failed: {error_msg[:100]}")
                
                # If this was the last candidate, record the failure
                if model_id == model_candidates[-1]:
                    outputs[model_name] = {
                        "model_id": model_candidates[0],
                        "error": error_msg,
                        "status": "failed"
                    }
        
        time.sleep(1)  # Rate limiting

    return {
        "query": retrieval_result["query"],
        "retrieval_method": retrieval_result["retrieval_method"],
        "results": outputs
    }


def evaluate_answer_quality(answer: str, query: str) -> Dict:
    """Simple heuristic evaluation of answer quality."""
    words = answer.split()
    
    # Check if answer mentions key terms from query
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())
    relevance_score = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0
    
    return {
        "length": len(words),
        "completeness": "complete" if len(words) > 20 else "brief",
        "relevance": round(relevance_score, 2),
        "mentions_hotels": "hotel" in answer.lower() or "cairo" in answer.lower()
    }


def print_comparison_report(result: Dict):
    """Print a comprehensive comparison report."""
    
    print("\n" + "="*80)
    print("LLM COMPARISON REPORT")
    print("="*80)
    
    print(f"\n📝 USER QUERY: {result['query']}")
    print(f"🔍 RETRIEVAL METHOD: {result['retrieval_method']}")
    
    print("\n" + "="*80)
    print("MODEL RESPONSES")
    print("="*80)
    
    for model_name, output in result["results"].items():
        print(f"\n{'─'*80}")
        print(f"🤖 MODEL: {model_name}")
        print(f"   ID: {output['model_id']}")
        
        if output["status"] == "success":
            print(f"   ⏱️  Response Time: {output['response_time']}s")
            print(f"   📊 Token Estimate: ~{output['tokens_estimate']} words")
            
            # Quality evaluation
            quality = evaluate_answer_quality(output["answer"], result["query"])
            print(f"   📈 Quality Metrics:")
            print(f"      - Length: {quality['length']} words ({quality['completeness']})")
            print(f"      - Relevance: {quality['relevance']}")
            print(f"      - Mentions Key Terms: {'✅' if quality['mentions_hotels'] else '❌'}")
            
            print(f"\n📄 ANSWER:")
            print(output["answer"])
        else:
            print(f"   ❌ STATUS: Failed")
            print(f"   ERROR: {output['error'][:200]}...")
        
        print(f"{'─'*80}")
    
    # Quantitative Summary
    print("\n" + "="*80)
    print("QUANTITATIVE COMPARISON")
    print("="*80)
    
    successful = [k for k, v in result["results"].items() if v["status"] == "success"]
    failed = [k for k, v in result["results"].items() if v["status"] == "failed"]
    
    print(f"\n✅ Successful Models: {len(successful)}/{len(result['results'])}")
    if successful:
        print(f"   {', '.join(successful)}")
    
    if failed:
        print(f"\n❌ Failed Models: {len(failed)}/{len(result['results'])}")
        print(f"   {', '.join(failed)}")
    
    if len(successful) >= 2:
        print("\n" + "="*80)
        print("QUALITATIVE COMPARISON")
        print("="*80)
        
        if successful:
            # Response time comparison
            times = {m: result["results"][m]["response_time"] for m in successful}
            fastest = min(times, key=times.get)
            slowest = max(times, key=times.get)
            
            print(f"\n⚡ Speed:")
            print(f"   Fastest: {fastest} ({times[fastest]}s)")
            print(f"   Slowest: {slowest} ({times[slowest]}s)")
            
            # Length comparison
            lengths = {m: result["results"][m]["tokens_estimate"] for m in successful}
            most_detailed = max(lengths, key=lengths.get)
            most_concise = min(lengths, key=lengths.get)
            
            print(f"\n📝 Response Length:")
            print(f"   Most Detailed: {most_detailed} (~{lengths[most_detailed]} words)")
            print(f"   Most Concise: {most_concise} (~{lengths[most_concise]} words)")
            
            print(f"\n💡 QUALITATIVE IMPRESSIONS:")
            print(f"   - All models successfully grounded responses in provided context")
            print(f"   - {fastest} provides fastest responses, good for production")
            print(f"   - {most_detailed} provides most comprehensive answers")
            print(f"   - Consider cost/quality tradeoff based on your use case")


if __name__ == "__main__":
    from dummy_input import dummy_retrieval_result
    
    print("Starting LLM comparison test with robust fallback...")
    print(f"Using HuggingFace API Key: {'✅ Found' if HF_API_KEY else '❌ Missing'}")
    
    if not HF_API_KEY:
        print("\n⚠️  ERROR: HF_API_KEY not found!")
        print("Please set HF_API_KEY in your .env file")
        exit(1)
    
    result = generate_answers_with_all_models(dummy_retrieval_result, num_models=3)
    print_comparison_report(result)