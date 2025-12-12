
import re
import json
from typing import Dict
from huggingface_hub import InferenceClient

# -----------------------------
# 1. INIT HF CLIENT
# -----------------------------
HF_TOKEN = "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" 
client = InferenceClient(
    model="google/gemma-2-2b-it",
    token=HF_TOKEN
)

# -----------------------------
# 2. BASIC TEXT CLEANING
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# -----------------------------
# 3. LLM INTENT CLASSIFICATION
# -----------------------------
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
9. unknown              → unclear or unrelated

Return ONLY a JSON object like this:

{{ "intent": "<intent>", "reason": "<brief explanation>" }}

User Query: "{cleaned_query}"
"""

    try:
        # Using chat_completion which returns an object
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        
        # Extract the message content from the response object
        # The response is a ChatCompletionOutput object with choices attribute
        if hasattr(response, 'choices') and len(response.choices) > 0:
            response_text = response.choices[0].message.content
        else:
            # Fallback: try to access directly
            response_text = str(response)
        
        # Extract JSON from response (LLM might add extra text)
        response_text = response_text.strip()
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
        else:
            # If no JSON found, fallback
            parsed = {"intent": "unknown", "reason": f"No valid JSON in response. Got: {response_text[:100]}..."}
        
        if "intent" not in parsed:
            parsed["intent"] = "unknown"
            
        return parsed

    except json.JSONDecodeError as e:
        return {"intent": "unknown", "reason": f"JSON decode error: {str(e)}"}
    except Exception as e:
        return {"intent": "unknown", "reason": f"Error: {str(e)}"}

# -----------------------------
# 4. PIPELINE WRAPPER
# -----------------------------
def detect_intent(user_query: str) -> Dict:
    return classify_intent_llm(user_query)

# -----------------------------
# 5. FALLBACK RULE-BASED CLASSIFIER
# -----------------------------
def classify_intent_rules(query: str) -> Dict:
    """Simple rule-based fallback classifier"""
    query_lower = query.lower()
    
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

# -----------------------------
# 6. HYBRID INTENT DETECTION
# -----------------------------
def hybrid_intent_detection(user_query: str) -> Dict:
    """Try LLM first, fallback to rules if LLM fails"""
    # Try LLM classification
    llm_result = classify_intent_llm(user_query)
    
    # Check if LLM returned a valid result (not unknown or error)
    if (llm_result.get("intent") != "unknown" and 
        "Error" not in llm_result.get("reason", "") and
        "No valid JSON" not in llm_result.get("reason", "")):
        llm_result["method"] = "llm"
        return llm_result
    
    # Fallback to rule-based
    rule_result = classify_intent_rules(user_query)
    return rule_result


"""
entities.py
-------------
Extracts relevant entities from user queries for the hotel domain
using a dataset (hotels.csv) instead of a fixed list.
"""

import spacy
import pandas as pd
from typing import Dict, List

# -----------------------------
# 1. LOAD SPACY MODEL
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 2. LOAD HOTEL DATASET
# -----------------------------
hotels_df = pd.read_csv("data/hotels.csv")  # adjust path if needed

# Assume your hotels.csv has columns: 'hotel_name', 'city', 'country'
HOTELS = hotels_df['hotel_name'].str.lower().tolist()
CITIES = hotels_df['city'].str.lower().tolist()
COUNTRIES = hotels_df['country'].str.lower().tolist()

N = hotels_df['hotel_name'].str.lower().tolist()
CITIES = hotels_df['city'].str.lower().tolist()
COUNTRIES = hotels_df['country'].str.lower().tolist()

TRAVELER_TYPES = ["family", "couple", "solo", "business", "group"]

# -----------------------------
# 3. ENTITY EXTRACTION FUNCTION
# -----------------------------
def extract_entities(query: str) -> Dict[str, List[str]]:
    query_lower = query.lower()
    doc = nlp(query_lower)

    entities = {
        "hotels": [],
        "cities": [],
        "countries": [],
        "traveler_types": [],
        "other": []
    }

    # 3a. Match hotels from dataset
    for hotel in HOTELS:
        if hotel in query_lower:
            entities["hotels"].append(hotel)

    # 3b. Match cities and countries from dataset
    for city in CITIES:
        if city in query_lower:
            entities["cities"].append(city)
    for country in COUNTRIES:
        if country in query_lower:
            entities["countries"].append(country)

    # 3c. Match traveler types
    for t in TRAVELER_TYPES:
        if t in query_lower:
            entities["traveler_types"].append(t)

    # 3d. Use spaCy NER as fallback for any other entities
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            if ent.text not in entities["cities"] and ent.text not in entities["countries"]:
                entities["other"].append(ent.text)
        elif ent.label_ == "ORG":
            if ent.text not in entities["hotels"]:
                entities["other"].append(ent.text)

    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))

    return entities

# -----------------------------
# 4. TEST
# -----------------------------
if __name__ == "__main__":
    test_queries = [
        "Find me a Marriott in Dubai for a family",
        "Show Hilton Cairo reviews",
        "Do Egyptians need a visa to travel to Thailand?",
        "Book a hotel in Paris for solo travelers"
    ]

    for q in test_queries:
        print(f"Query: {q}\nEntities: {extract_entities(q)}\n")

# -----------------------------
# 7. MAIN FUNCTION
# -----------------------------
# def main():
#     test_queries = [
#         "Find me hotels in Dubai under $200",
#         "Does the Marriott Downtown have a swimming pool?",
#         "Show me reviews for Hilton Cairo",
#         "What is the distance between my hotel and the pyramids?",
#         "Do Egyptians need a visa to travel to Thailand?",
#         "I need a hotel with free wifi and breakfast",
#         "Which hotels are closest to the airport?",
#         "Tell me about the best luxury hotels in Paris"
#     ]

#     print("Intent Classification Results:\n")
#     for q in test_queries:
#         result = hybrid_intent_detection(q)
#         print(f"Query: {q}")
#         print(f"Intent: {result.get('intent', 'unknown')}")
#         print(f"Reason: {result.get('reason', 'No reason provided')}")
#         print(f"Method: {result.get('method', 'unknown')}")
#         print("-" * 50 + "\n")

# # -----------------------------
# # Run main
# # -----------------------------
# if __name__ == "__main__":
#     main()