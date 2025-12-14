import re
import json
from typing import Dict, Any
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HF_TOKEN
)

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def classify_intent_llm(query: str, max_tokens=200) -> Dict:
    cleaned_query = clean_text(query)
    
    prompt = f"""You are an intent classifier for a hotel search and travel system.
Classify the user's query into one of these intents:

1. hotel_search         → search/filter hotels (price, stars, area)
2. hotel_details        → ask about a specific hotel
3. hotel_reviews        → want reviews or ratings
4. hotel_recommendation → best options or suggestions
5. amenity_filtering    → filter by amenities (wifi, pool, spa, parking)
6. location_query       → distances, nearby places, neighborhoods
7. visa_requirements    → visa rules for traveling to a country
8. general_question     → general hotel/travel info
9. score_filtering      → filter by scores (overall, value, location, staff, facilities)
10. unknown             → unclear or unrelated

Return ONLY a JSON object like this:

{{ "intent": "<intent>", "reason": "<brief explanation>" }}

User Query: "{cleaned_query}"
"""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_text = response.choices[0].message.content
        else:
            response_text = str(response)
        
        response_text = response_text.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
        else:
            parsed = {"intent": "unknown", "reason": f"No valid JSON in response. Got: {response_text[:100]}..."}
        
        if "intent" not in parsed:
            parsed["intent"] = "unknown"
            
        return parsed

    except json.JSONDecodeError as e:
        return {"intent": "unknown", "reason": f"JSON decode error: {str(e)}"}
    except Exception as e:
        return {"intent": "unknown", "reason": f"Error: {str(e)}"}

def classify_intent_rules(query: str) -> Dict:
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in ["score", "rating between", "average score", "avg score", "value for money", "location score", "staff score", "best staff", "best value"]):
        return {
            "intent": "score_filtering",
            "reason": "Query contains score filtering keywords",
            "method": "rule_based"
        }
    
    intent_keywords = {
        "hotel_search": ["find", "search", "look for", "show me", "need", "want", "looking"],
        "hotel_details": ["does", "has", "have", "what is", "tell me about", "details", "information"],
        "hotel_reviews": ["review", "rating", "rate", "how good", "feedback", "rated"],
        "location_query": ["distance", "near", "close to", "far from", "how far", "location", "where"],
        "visa_requirements": ["visa", "passport", "entry requirement", "travel document"],
        "amenity_filtering": ["pool", "wifi", "parking", "gym", "spa", "breakfast", "amenity", "facility"],
        "hotel_recommendation": ["best", "recommend", "suggest", "good", "top", "popular"],
        "general_question": ["what", "how", "why", "when", "which", "can you explain"],
    }
    
    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                return {
                    "intent": intent,
                    "reason": f"Matched keyword: '{keyword}'",
                    "method": "rule_based"
                }
    
    return {"intent": "unknown", "reason": "No keywords matched", "method": "rule_based"}

def hybrid_intent_detection(query: str) -> Dict:
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in ["score", "rating between", "average score", "avg score", "value for money", "location score", "staff score", "best staff", "best value"]):
        return {"intent": "score_filtering", "reason": "Query contains score filtering keywords", "method": "rule_based"}
    
    visa_keywords = ["visa", "passport", "entry requirement", "travel document", "need a visa", "require a visa", "visa requirement"]
    
    if any(keyword in query_lower for keyword in visa_keywords):
        return {"intent": "visa_requirements", "reason": "Query contains visa-related keywords", "method": "rule_based_visa"}
    
    llm_result = classify_intent_llm(query)
    
    if (llm_result.get("intent") != "unknown" and 
        "Error" not in llm_result.get("reason", "") and
        "No valid JSON" not in llm_result.get("reason", "")):
        llm_result["method"] = "llm"
        return llm_result
    
    rule_result = classify_intent_rules(query)
    return rule_result

def refine_intent_with_entities(raw_intent: str, entities: Dict[str, Any], query: str) -> str:
    intent = raw_intent or "unknown"
    q = query.lower()

    if entities.get("min_score") is not None or entities.get("max_score") is not None:
        return "hotels_by_score_range"
    
    if "value for money" in q or "best value" in q:
        return "best_value_hotels"
    
    if "location score" in q:
        return "hotels_by_location_score"
    
    if "staff" in q and ("best staff" in q or "staff score" in q):
        return "hotels_with_best_staff"

    if entities.get("countries") and not entities.get("cities"):
        return "hotel_search"
    
    if entities.get("gender") and (entities.get("age_numbers") or entities.get("age")):
        return "hotels_by_traveler_gender_age"

    if entities.get("min_facility_score") is not None or entities.get("facilities") or entities.get("facility_ratings"):
        if entities.get("min_facility_score") is None:
            m = re.search(r'(?:facilities|facility|facility score|facilities score)\s*(?:>=|>|>=|at least|above|over|:)?\s*([0-9]+(?:\.[0-9]+)?)', q)
            if not m:
                m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:\+)?\s*(?:facilities|facility)', q)
            if m:
                try:
                    entities["min_facility_score"] = float(m.group(1))
                    entities.setdefault("facility_ratings", {})["facilities"] = entities["min_facility_score"]
                except Exception:
                    pass
        return "amenity_filtering"

    if entities.get("traveler_types"):
        return "traveller_query"

    patterns = [
        r'(?:at\s+least|minimum|min|>=|greater\s+than|more\s+than|over)\s+(\d+)\s*(?:reviews|review)',
        r'(\d+)\s*(?:\+|plus)\s*(?:reviews|review)',
        r'(\d+)\s*(?:or\s+more)\s*(?:reviews|review)',
        r'(\d+)\s*(?:reviews|review)'
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            try:
                entities["min_reviews"] = int(m.group(1))
                return "hotels_with_min_reviews"
            except Exception:
                pass

    if intent == "hotel_reviews":
        return "hotel_reviews"

    return intent

def main():
    test_queries = [
        "Find me hotels in Dubai under $200",
        "Does the Marriott Downtown have a swimming pool?",
        "Show me reviews for Hilton Cairo",
        "What is the distance between my hotel and the pyramids?",
        "Do Egyptians need a visa to travel to Thailand?",
        "I need a hotel with free wifi and breakfast",
        "Which hotels are closest to the airport?",
        "Tell me about the best luxury hotels in Paris"
    ]

    print("\n" + "="*70)
    print("INTENT CLASSIFICATION TEST")
    print("="*70 + "\n")
    
    for q in test_queries:
        result = hybrid_intent_detection(q)
        print(f"Query: {q}")
        print(f"  Intent: {result.get('intent', 'unknown')}")
        print(f"  Reason: {result.get('reason', 'No reason provided')}")
        print(f"  Method: {result.get('method', 'unknown')}")
        print("-" * 70)

if __name__ == "__main__":
    main()