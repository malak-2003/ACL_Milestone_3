import pandas as pd
from pathlib import Path
import spacy
import re
from typing import Dict, List

# -----------------------------
# 1. LOAD HOTEL DATASET
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  
DATA_PATH = BASE_DIR / "data" / "hotels.csv"
hotels_df = pd.read_csv(DATA_PATH) 
HOTELS = hotels_df['hotel_name'].str.lower().tolist()

# -----------------------------
# 2. LOAD SPACY MODEL
# -----------------------------
# Small English model, offline
nlp = spacy.load("en_core_web_sm") 

# -----------------------------
# 3. PREDEFINED TRAVELER TYPES
# -----------------------------
TRAVELER_TYPES = ["family", "couple", "solo", "business"]

# FIXED: Better gender patterns with word boundaries
GENDERS = {
    'male': [
        r'\bmale\b', r'\bman\b', r'\bmen\b', r'\bboy\b', r'\bboys\b',
        r'\bgentleman\b', r'\bgentlemen\b', r'\bguy\b', r'\bguys\b'
    ],
    'female': [
        r'\bfemale\b', r'\bwoman\b', r'\bwomen\b', r'\bgirl\b', r'\bgirls\b',
        r'\blady\b', r'\bladies\b', r'\bgal\b', r'\bgals\b'
    ]
}

AGE_PATTERNS = [
    (r'(\d+)\s*years?(\s*old)?', 'exact_age'),
    (r'(\d+)[-+]', 'age_range_start'),
    (r'aged\s*(\d+)', 'exact_age'),
    (r'age\s*(\d+)', 'exact_age'),
    (r'(\d+)-(\d+)\s*years?', 'age_range'),
    (r'(\d+)s', 'decade_age'),  # e.g., 20s, 30s
    (r'(\d+)\s*to\s*(\d+)\s*years?', 'age_range_to'),
    (r'(\d+)\s*and\s*(\d+)\s*years?', 'age_range_and'),
    (r'child', 'age_group'),
    (r'teenager|teen', 'age_group'),
    (r'adult', 'age_group'),
    (r'senior|elderly', 'age_group'),
    (r'infant|baby|toddler', 'age_group'),
    (r'young adult', 'age_group'),
    (r'middle aged', 'age_group'),
    (r'(\d+)\s*month', 'months'),
    (r'(\d+)\s*week', 'weeks')
]

# -----------------------------
# 4. FIXED GENDER EXTRACTION FUNCTION
# -----------------------------
def extract_gender_entities(query: str) -> List[str]:
    """
    Extract gender entities from query using regex with word boundaries
    Returns list of genders (male, female)
    """
    genders_found = []
    query_lower = query.lower()
    
    # Check for female patterns first (to avoid "man" in "woman" matching)
    for gender, patterns in GENDERS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                if gender not in genders_found:
                    genders_found.append(gender)
    
    return genders_found

# -----------------------------
# 5. SIMPLIFIED AGE EXTRACTION FUNCTION
# -----------------------------
def extract_age_numbers(query: str) -> List[int]:
    """
    Extract age numbers from query
    Returns list of integers (single age or range ages)
    """
    age_numbers = []
    query_lower = query.lower()
    
    # Pattern for exact ages (e.g., "25 years old", "age 35")
    exact_patterns = [
        r'(\d+)\s*years?\s*old',
        r'aged\s*(\d+)',
        r'age\s*(\d+)',
        r'(\d+)\s*years?',
    ]
    
    # Pattern for age ranges (e.g., "5-10 years", "18 to 25")
    range_patterns = [
        r'(\d+)-(\d+)\s*years?',
        r'(\d+)\s*to\s*(\d+)\s*years?',
        r'(\d+)\s*and\s*(\d+)\s*years?',
    ]
    
    # Pattern for decades (e.g., "30s", "40s")
    decade_patterns = [
        r'(\d+)s',
    ]
    
    # Pattern for age ranges with plus/minus (e.g., "25+", "65+")
    plus_minus_patterns = [
        r'(\d+)\+',
        r'(\d+)-',  # For patterns like "18-"
    ]
    
    # First check for ranges
    for pattern in range_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            if isinstance(match, tuple):
                # For patterns with two numbers (e.g., "5-10")
                age1, age2 = match[0], match[1]
                age_numbers.extend([int(age1), int(age2)])
            else:
                age_numbers.append(int(match))
    
    # Check for exact ages
    for pattern in exact_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            if isinstance(match, tuple):
                age_numbers.append(int(match[0]))
            else:
                age_numbers.append(int(match))
    
    # Check for decades
    for pattern in decade_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            age_numbers.append(int(match))
    
    # Check for plus/minus patterns
    for pattern in plus_minus_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            age_numbers.append(int(match))
    
    # Remove duplicates and sort
    age_numbers = sorted(list(set(age_numbers)))
    
    return age_numbers

def extract_age_entities_simple(query: str) -> List[Dict]:
    """
    Simple age extraction - returns age numbers only
    """
    age_numbers = extract_age_numbers(query)
    age_entities = []
    
    for age in age_numbers:
        age_entities.append({
            "value": age,
            "type": "age_number"
        })
    
    return age_entities

# -----------------------------
# 6. FIXED ENTITY EXTRACTION FUNCTION
# -----------------------------
def extract_entities(query: str) -> Dict[str, List]:
    query_lower = query.lower()
    doc = nlp(query)

    entities = {
        "hotels": [],
        "cities": [],
        "countries": [],
        "traveler_types": [],
        "nationality": [],
        "age": [],  # Now stores age numbers only
        "gender": [],
        "age_numbers": []  # New: stores just the numbers
    }

    # Match hotels from dataset
    for hotel in HOTELS:
        if hotel in query_lower:
            entities["hotels"].append(hotel)

    # Match traveler types
    for t in TRAVELER_TYPES:
        if t in query_lower:
            entities["traveler_types"].append(t)

    # FIXED: Use the new gender extraction function
    entities["gender"] = extract_gender_entities(query)

    # Extract age numbers (simplified)
    age_numbers = extract_age_numbers(query)
    entities["age_numbers"] = age_numbers
    
    # For backward compatibility, also add to age list as strings
    entities["age"] = [str(age) for age in age_numbers]

    # Use spaCy NER to extract cities, countries, and nationalities
    for ent in doc.ents:
        if ent.label_ == "GPE":  # Geopolitical entity (city, country)
            text = ent.text.lower()
            # Check if it's a known city/country name
            if text not in entities["cities"] and text not in entities["countries"]:
                # Simple heuristic: assume it could be both city and country
                # You could add more sophisticated logic here
                entities["cities"].append(text)
                entities["countries"].append(text)
        elif ent.label_ == "NORP":  # Nationalities or religious or political groups
            if ent.text.lower() not in entities["nationality"]:
                entities["nationality"].append(ent.text)
        elif ent.label_ == "LOC":  # Locations (mountains, lakes, etc.)
            # Could be tourist attractions
            if "hotel" not in ent.text.lower() and "resort" not in ent.text.lower():
                entities["cities"].append(ent.text.lower())
    
    # Remove duplicates from simple lists
    for key in ["hotels", "cities", "countries", "traveler_types", "nationality", "age", "gender", "age_numbers"]:
        if key in entities:
            if key == "age_numbers":
                # For age numbers, keep as integers and sort
                entities[key] = sorted(list(set(entities[key])))
            else:
                entities[key] = list(set(entities[key]))

    return entities

# -----------------------------
# 7. FORMATTED OUTPUT FUNCTION
# -----------------------------
def format_entities_output(entities: Dict) -> str:
    """Create a nicely formatted string of extracted entities"""
    output = []
    
    if entities["hotels"]:
        output.append(f"🏨 Hotels: {', '.join(entities['hotels'])}")
    
    if entities["cities"]:
        output.append(f"📍 Cities: {', '.join(entities['cities'])}")
    
    if entities["countries"]:
        output.append(f"🌍 Countries: {', '.join(entities['countries'])}")
    
    if entities["traveler_types"]:
        output.append(f"👥 Traveler Types: {', '.join(entities['traveler_types'])}")
    
    if entities["nationality"]:
        output.append(f"🎌 Nationalities: {', '.join(entities['nationality'])}")
    
    if entities["gender"]:
        output.append(f"🚻 Gender: {', '.join(entities['gender'])}")
    
    if entities["age_numbers"]:
        if len(entities["age_numbers"]) == 1:
            output.append(f"🎂 Age: {entities['age_numbers'][0]} years")
        elif len(entities["age_numbers"]) == 2:
            output.append(f"🎂 Age Range: {entities['age_numbers'][0]}-{entities['age_numbers'][1]} years")
        else:
            output.append(f"🎂 Ages: {', '.join(map(str, entities['age_numbers']))} years")
    
    return "\n".join(output) if output else "No entities detected"

# -----------------------------
# 8. TEST WITH FIXED GENDER EXTRACTION
# -----------------------------
if __name__ == "__main__":
    # Focus on gender extraction test cases
    gender_test_queries = [
        "Book accommodation in Istanbul for a business woman aged 35",
        "Find hotels in Paris for male travelers",
        "Looking for resorts in Maldives for a couple (man and woman)",
        "Search for family hotels in London",
        "Need business hotels in Dubai for female executives",
        "Find solo male traveler accommodation in Tokyo",
        "Book hotel in New York for a gentleman",
        "Search for hotels in Sydney for ladies",
        "Find accommodation in Berlin for guys on business trip",
        "Book resort in Bali for girls trip"
    ]
    
    print("=" * 100)
    print("GENDER EXTRACTION TESTING - FIXED VERSION")
    print("=" * 100)
    
    for i, q in enumerate(gender_test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i:02d}: {q}")
        print(f"{'='*80}")
        
        entities = extract_entities(q)
        
        # Show gender extraction specifically
        print(f"\n📊 Gender Extraction:")
        print(f"  Query contains 'man': {'man' in q.lower()}")
        print(f"  Query contains 'woman': {'woman' in q.lower()}")
        print(f"  Extracted genders: {entities['gender']}")
        
        formatted_output = format_entities_output(entities)
        print(f"\n📋 All Entities:")
        print(formatted_output)
    
    # Original test queries for completeness
    print(f"\n{'='*100}")
    print("ORIGINAL TEST QUERIES - AGE EXTRACTION")
    print(f"{'='*100}")
    
    test_queries = [
        "Find me a Han River Oasis in Dubai for a family with children aged 5-10 years for women",
        "Show Hilton Cairo reviews for solo female travelers in their 30s",
        "Do Egyptians aged 25+ need a visa to travel to France?",
        "Book a Sheraton for a business trip in Paris for adults only",
        "I want a Ritz-Carlton for a couple in Tokyo with kids aged 3 and 5",
    ]
    
    for i, q in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i:02d}: {q}")
        print(f"{'='*80}")
        
        entities = extract_entities(q)
        formatted_output = format_entities_output(entities)
        
        print("\n" + formatted_output)
    
    print(f"\n{'='*100}")
    print(f"DATASET STATISTICS:")
    print(f"{'='*100}")
    print(f"Hotels in dataset: {len(HOTELS)}")
    print(f"Sample hotels: {HOTELS[:5] if HOTELS else 'No hotels found in dataset'}")


























# """
# entities.py
# -------------
# Extracts relevant entities from user queries for the hotel domain
# using a dataset (hotels.csv) instead of a fixed list.
# """

# import spacy
# import pandas as pd
# from typing import Dict, List

# # -----------------------------
# # 1. LOAD SPACY MODEL
# # -----------------------------
# nlp = spacy.load("en_core_web_sm")

# # -----------------------------
# # 2. LOAD HOTEL DATASET
# # -----------------------------
# hotels_df = pd.read_csv("data/hotels.csv")  # adjust path if needed

# # Assume your hotels.csv has columns: 'hotel_name', 'city', 'country'
# HOTELS = hotels_df['hotel_name'].str.lower().tolist()
# CITIES = hotels_df['city'].str.lower().tolist()
# COUNTRIES = hotels_df['country'].str.lower().tolist()

# N = hotels_df['hotel_name'].str.lower().tolist()
# CITIES = hotels_df['city'].str.lower().tolist()
# COUNTRIES = hotels_df['country'].str.lower().tolist()

# TRAVELER_TYPES = ["family", "couple", "solo", "business", "group"]

# # -----------------------------
# # 3. ENTITY EXTRACTION FUNCTION
# # -----------------------------
# def extract_entities(query: str) -> Dict[str, List[str]]:
#     query_lower = query.lower()
#     doc = nlp(query_lower)

#     entities = {
#         "hotels": [],
#         "cities": [],
#         "countries": [],
#         "traveler_types": [],
#         "other": []
#     }

#     # 3a. Match hotels from dataset
#     for hotel in HOTELS:
#         if hotel in query_lower:
#             entities["hotels"].append(hotel)

#     # 3b. Match cities and countries from dataset
#     for city in CITIES:
#         if city in query_lower:
#             entities["cities"].append(city)
#     for country in COUNTRIES:
#         if country in query_lower:
#             entities["countries"].append(country)

#     # 3c. Match traveler types
#     for t in TRAVELER_TYPES:
#         if t in query_lower:
#             entities["traveler_types"].append(t)

#     # 3d. Use spaCy NER as fallback for any other entities
#     for ent in doc.ents:
#         if ent.label_ in ["GPE", "LOC"]:
#             if ent.text not in entities["cities"] and ent.text not in entities["countries"]:
#                 entities["other"].append(ent.text)
#         elif ent.label_ == "ORG":
#             if ent.text not in entities["hotels"]:
#                 entities["other"].append(ent.text)

#     # Remove duplicates
#     for key in entities:
#         entities[key] = list(set(entities[key]))

#     return entities

# # -----------------------------
# # 4. TEST
# # -----------------------------
# if __name__ == "__main__":
#     test_queries = [
#         "Find me a Marriott in Dubai for a family",
#         "Show Hilton Cairo reviews",
#         "Do Egyptians need a visa to travel to Thailand?",
#         "Book a hotel in Paris for solo travelers"
#     ]

#     for q in test_queries:
#         print(f"Query: {q}\nEntities: {extract_entities(q)}\n")

# # -----------------------------
# # 7. MAIN FUNCTION
# # -----------------------------
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