import pandas as pd
from pathlib import Path
import spacy
import re
from typing import Dict, List, Optional, Tuple
import json

class EntityExtractorConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  
    DATA_PATH = BASE_DIR / "data" / "hotels.csv"
    TRAVELER_TYPES = ["family", "couple", "solo", "business", "group", "friends"]
    FACILITIES = ["cleanliness", "comfort", "facilities", "location", "staff", "value for money", "value"]
    GENDERS = {
        'male': [r'\bmale\b', r'\bman\b', r'\bmen\b', r'\bboy\b', r'\bboys\b', r'\bgentleman\b', r'\bgentlemen\b', r'\bguy\b', r'\bguys\b'],
        'female': [r'\bfemale\b', r'\bwoman\b', r'\bwomen\b', r'\bgirl\b', r'\bgirls\b', r'\blady\b', r'\bladies\b', r'\bgal\b', r'\bgals\b']
    }

class EnhancedEntityExtractor:
    def __init__(self, data_path: Optional[str] = None, spacy_model: str = "en_core_web_sm"):
        self.config = EntityExtractorConfig()
        self._load_hotel_data(data_path)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.nlp = None
    
    def _load_hotel_data(self, data_path: Optional[str] = None):
        try:
            path = Path(data_path) if data_path else self.config.DATA_PATH
            hotels_df = pd.read_csv(path)
            self.hotels = hotels_df['hotel_name'].str.lower().tolist()
            self.cities = hotels_df['city'].str.lower().dropna().unique().tolist()
            self.countries = hotels_df['country'].str.lower().dropna().unique().tolist()
        except FileNotFoundError:
            self.hotels = []
            self.cities = []
            self.countries = []
        except KeyError as e:
            self.hotels = []
            self.cities = []
            self.countries = []
    
    def extract_all(self, query: str) -> Dict[str, any]:
        query_lower = query.lower()
        entities = {
            "hotels": self._extract_hotels(query_lower),
            "cities": self._extract_cities_from_csv(query_lower),
            "countries": self._extract_countries_from_csv(query_lower),
            "traveler_types": self._extract_traveler_types(query_lower),
            "facilities": self._extract_facilities(query_lower),
            "facility_ratings": self._extract_facility_ratings(query),
            "nationality": [],
            "age_numbers": self._extract_age_numbers(query_lower),
            "gender": self._extract_gender(query_lower),
            "star_rating": self._extract_star_rating(query_lower),
            "min_rating": self._extract_min_rating(query_lower),
            "min_facility_score": None,
            "min_reviews": None
        }
        
        if self.nlp:
            spacy_entities = self._extract_spacy_entities(query)
            entities["nationality"].extend(spacy_entities["nationality"])
        
        for key in ["hotels", "cities", "countries", "traveler_types", "facilities", "nationality", "gender"]:
            entities[key] = list(set(entities[key]))
        
        entities["age_numbers"] = sorted(entities["age_numbers"])
        entities = self._enrich_entities_with_query_patterns(query, entities)
        
        return entities
    
    def _extract_hotels(self, query: str) -> List[str]:
        found_hotels = []
        for hotel in self.hotels:
            if hotel in query:
                found_hotels.append(hotel)
        query_words = query.split()
        for hotel in self.hotels:
            hotel_words = hotel.split()
            for i in range(len(query_words) - len(hotel_words) + 1):
                if ' '.join(query_words[i:i+len(hotel_words)]) == hotel:
                    found_hotels.append(hotel)
        return list(set(found_hotels))
    
    def _extract_cities_from_csv(self, query: str) -> List[str]:
        found_cities = []
        if not hasattr(self, 'cities') or not self.cities:
            return found_cities
        for city in self.cities:
            if city in query:
                found_cities.append(city)
        return list(set(found_cities))
    
    def _extract_countries_from_csv(self, query: str) -> List[str]:
        found_countries = []
        if not hasattr(self, 'countries') or not self.countries:
            return found_countries
        for country in self.countries:
            if country in query:
                found_countries.append(country)
        return list(set(found_countries))
    
    def _extract_traveler_types(self, query: str) -> List[str]:
        found_types = []
        for ttype in self.config.TRAVELER_TYPES:
            if ttype in query:
                found_types.append(ttype)
        return found_types
    
    def _extract_facilities(self, query: str) -> List[str]:
        found_facilities = []
        for facility in self.config.FACILITIES:
            if facility in query:
                normalized = "value" if "value" in facility else facility
                if normalized not in found_facilities:
                    found_facilities.append(normalized)
        return found_facilities
    
    def _extract_facility_ratings(self, query: str) -> Dict[str, float]:
        facility_ratings = {}
        query_lower = query.lower()
        facilities = ['cleanliness', 'comfort', 'facilities', 'location', 'staff', 'value']
        age_numbers = self._extract_age_numbers(query_lower)
        patterns = [
            (r'\b({facility})\s+(?:rating|score|level)\s+(?:of\s+)?(?:at least\s+)?(\d+(?:\.\d+)?)\b', 'direct_with_keyword'),
            (r'\b({facility})\s+(?:above|over|greater than|more than|at least|minimum|>=|>)\s*(\d+(?:\.\d+)?)\b', 'comparison'),
            (r'\b({facility})\s+(\d+(?:\.\d+)?)\+\b', 'with_plus'),
            (r'\b(?:rating|score)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s+(?:for|on|in)\s+({facility})\b', 'rating_first'),
            (r'\b(?:with|has|having)\s+({facility})\s+(?:of\s+)?(\d+(?:\.\d+)?)\b', 'with_preposition'),
            (r'\b({facility})\s*:\s*(\d+(?:\.\d+)?)\b', 'with_colon'),
            (r'\b({facility})\s*=\s*(\d+(?:\.\d+)?)\b', 'with_equals'),
        ]
        
        for facility in facilities:
            for pattern_template, pattern_type in patterns:
                pattern = pattern_template.replace('{facility}', facility)
                if facility == 'value':
                    pattern_variants = [pattern, pattern.replace('value', 'value for money')]
                else:
                    pattern_variants = [pattern]
                for p in pattern_variants:
                    matches = re.finditer(p, query_lower)
                    for match in matches:
                        groups = match.groups()
                        rating = None
                        if pattern_type in ['direct_with_keyword', 'comparison', 'with_plus', 'with_preposition', 'with_colon', 'with_equals']:
                            if groups[1] and groups[1].replace('.', '').isdigit():
                                rating = float(groups[1])
                        elif pattern_type == 'rating_first':
                            if groups[0] and groups[0].replace('.', '').isdigit():
                                rating = float(groups[0])
                        if rating is not None:
                            is_likely_age = False
                            if int(rating) in age_numbers:
                                context_start = max(0, match.start() - 10)
                                context_end = min(len(query_lower), match.end() + 10)
                                context = query_lower[context_start:context_end]
                                age_keywords = ['age', 'aged', 'years', 'old', 'year old', 'yr']
                                facility_keywords = ['rating', 'score', 'level', 'above', 'over', 'minimum']
                                age_keyword_nearby = any(keyword in context for keyword in age_keywords)
                                facility_keyword_nearby = any(keyword in context for keyword in facility_keywords)
                                if age_keyword_nearby and not facility_keyword_nearby:
                                    is_likely_age = True
                            if not is_likely_age and 0 <= rating <= 10:
                                norm_facility = 'value' if facility == 'value for money' else facility
                                facility_ratings[norm_facility] = rating
        
        list_pattern = r'\b(\d+(?:\.\d+)?)\s*,\s*(cleanliness|comfort|facilities|location|staff|value(?: for money)?)\b'
        list_matches = re.findall(list_pattern, query_lower)
        for rating_str, facility in list_matches:
            if rating_str.replace('.', '').isdigit():
                rating = float(rating_str)
                if int(rating) not in age_numbers and 0 <= rating <= 10:
                    norm_facility = 'value' if 'value' in facility else facility
                    facility_ratings[norm_facility] = rating
        
        return facility_ratings
    
    def _extract_gender(self, query: str) -> List[str]:
        query_lower = query.lower()
        
        gender_patterns = [
            r'\b(male|female)\b',
            r'\b(man|woman|men|women)\b',
        ]
        
        genders = []
        for pattern in gender_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if match in ['male', 'man', 'men']:
                    genders.append('Male')
                elif match in ['female', 'woman', 'women']:
                    genders.append('Female')
        
        return list(set(genders))

    def _extract_age_numbers(self, query: str) -> List[int]:
        query_lower = query.lower()
        
        age_patterns = [
            r'\baged?\s+(\d+)',
            r'\b(\d+)\s+years?\s+old',
            r'\bin\s+their\s+(\d+)s?',
        ]
        
        ages = []
        for pattern in age_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                try:
                    age = int(match)
                    if 0 < age < 120:
                        ages.append(age)
                except:
                    continue
        
        return list(set(ages))

    def map_entities_to_params(self, intent: str, entities: Dict[str, any], query: str) -> Dict[str, any]:
        params = {}
        params["limit"] = entities.get("limit", 5)
        
        if intent == "hotels_by_traveler_gender_age":
            if entities.get("gender"):
                params["gender"] = entities["gender"][0]
            
            if entities.get("age_numbers"):
                params["age"] = entities["age_numbers"][0]
            elif entities.get("age"):
                try:
                    params["age"] = int(entities["age"])
                except:
                    pass
            
            return params
        
        if entities.get("cities"):
            params["city"] = entities["cities"][0].title()
        
        if entities.get("countries"):
            params["country"] = entities["countries"][0].title()
        
        if entities.get("hotels"):
            params["hotel_name"] = entities["hotels"][0]
        
        if entities.get("traveler_types"):
            params["type"] = entities["traveler_types"][0]
        
        if entities.get("min_rating") is not None:
            params["min_rating"] = entities["min_rating"]
        
        if entities.get("min_facility_score") is not None:
            params["min_facility_score"] = entities["min_facility_score"]
        
        if entities.get("min_reviews") is not None:
            params["min_reviews"] = entities["min_reviews"]
        
        if entities.get("min_score") is not None:
            params["min_score"] = entities["min_score"]
        
        if entities.get("max_score") is not None:
            params["max_score"] = entities["max_score"]
        
        if entities.get("min_location_score") is not None:
            params["min_location_score"] = entities["min_location_score"]
        
        return params
    
    def _extract_star_rating(self, query: str) -> Optional[int]:
        patterns = [
            r'\b(\d)-star\b',
            r'\b(\d)\s+star\b',
            r'\b(five|four|three|two|one)\s+star\b'
        ]
        number_words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                rating_str = match.group(1)
                if rating_str.isdigit():
                    return int(rating_str)
                elif rating_str in number_words:
                    return number_words[rating_str]
        return None
    
    def _extract_min_rating(self, query: str) -> Optional[float]:
        patterns = [
            r'\brating\s+(?:above|over|>=|>)\s*(\d+(?:\.\d+)?)\b',
            r'\b(\d+(?:\.\d+)?)\s*\+\b',
            r'\bminimum\s+rating\s+(?:of\s+)?(\d+(?:\.\d+)?)\b',
            r'\b(?:rating|score)\s+(?:of\s+)?(?:at least\s+)?(\d+(?:\.\d+)?)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                rating = float(match.group(1))
                if 0 <= rating <= 10:
                    return rating
        return None
    
    def _extract_spacy_entities(self, query: str) -> Dict[str, List[str]]:
        entities = {"nationality": []}
        if not self.nlp:
            return entities
        doc = self.nlp(query)
        for ent in doc.ents:
            if ent.label_ == "NORP":
                entities["nationality"].append(ent.text.lower())
        return entities
    
    def _enrich_entities_with_query_patterns(self, query: str, entities: Dict[str, any]) -> Dict[str, any]:
        q_lower = query.lower()
        
        m = re.search(r'(?:facilities|facility|facility score|facilities score)\s*(?:>=|>|at least|above|over|:)?\s*([0-9]+(?:\.[0-9]+)?)', q_lower)
        if not m:
            m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:\+)?\s*(?:facilities|facility)', q_lower)
        if m:
            try:
                entities["min_facility_score"] = float(m.group(1))
            except Exception:
                pass
        
        if entities.get("min_facility_score") is None and entities.get("facility_ratings"):
            vals = list(entities["facility_ratings"].values())
            if vals:
                entities["min_facility_score"] = max(vals)
        
        patterns = [
            r'(?:at\s+least|minimum|min|>=|greater\s+than|more\s+than|over)\s+(\d+)\s*(?:reviews|review)',
            r'(\d+)\s*(?:\+|plus)\s*(?:reviews|review)',
            r'(\d+)\s*(?:or\s+more)\s*(?:reviews|review)',
            r'(\d+)\s*(?:reviews|review)'
        ]
        found = False
        for p in patterns:
            mm = re.search(p, q_lower)
            if mm:
                try:
                    entities["min_reviews"] = int(mm.group(1))
                    found = True
                    break
                except Exception:
                    pass
        
        if not found:
            number_words = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
                "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
                "nineteen": 19, "twenty": 20
            }
            for word, val in number_words.items():
                if re.search(r'\b' + re.escape(word) + r'\b', q_lower):
                    entities["min_reviews"] = val
                    found = True
                    break
        
        range_match = re.search(r'between\s+([0-9]+(?:\.[0-9]+)?)\s+and\s+([0-9]+(?:\.[0-9]+)?)', q_lower)
        if range_match:
            entities["min_score"] = float(range_match.group(1))
            entities["max_score"] = float(range_match.group(2))
        
        loc_match = re.search(r'location\s+score\s+(?:above|over|>=|>)\s*([0-9]+(?:\.[0-9]+)?)', q_lower)
        if loc_match:
            entities["min_location_score"] = float(loc_match.group(1))
        
        if "limit" not in entities:
            limit_match = re.search(r'\blimit\s+(\d+)\b', q_lower)
            if limit_match:
                entities["limit"] = int(limit_match.group(1))
            else:
                entities["limit"] = 5
        
        return entities
    
    def nationality_to_country(self, nationality: str) -> str:
        return {
            "indian": "India", "egyptian": "Egypt", "american": "United States",
            "british": "United Kingdom",
        }.get(nationality.lower().strip(), nationality.title())

def format_entities_output(entities: Dict) -> str:
    output = []
    if entities.get("hotels"):
        output.append(f"Hotels: {', '.join(entities['hotels'])}")
    if entities.get("cities"):
        output.append(f"Cities: {', '.join(entities['cities'])}")
    if entities.get("countries"):
        output.append(f"Countries: {', '.join(entities['countries'])}")
    if entities.get("traveler_types"):
        output.append(f"Traveler Types: {', '.join(entities['traveler_types'])}")
    if entities.get("facilities"):
        output.append(f"Facilities: {', '.join(entities['facilities'])}")
    if entities.get("facility_ratings"):
        ratings_str = ', '.join([f"{k}: {v}" for k, v in entities['facility_ratings'].items()])
        output.append(f"Facility Ratings: {ratings_str}")
    if entities.get("star_rating"):
        output.append(f"Star Rating: {entities['star_rating']} stars")
    if entities.get("min_rating"):
        output.append(f"Min Overall Rating: {entities['min_rating']}")
    if entities.get("nationality"):
        output.append(f"Nationalities: {', '.join(entities['nationality'])}")
    if entities.get("gender"):
        output.append(f"Gender: {', '.join(entities['gender'])}")
    if entities.get("age_numbers"):
        if len(entities["age_numbers"]) == 1:
            output.append(f"Age: {entities['age_numbers'][0]} years")
        elif len(entities["age_numbers"]) == 2:
            output.append(f"Age Range: {entities['age_numbers'][0]}-{entities['age_numbers'][1]} years")
        else:
            output.append(f"Ages: {', '.join(map(str, entities['age_numbers']))} years")
    if entities.get("min_facility_score"):
        output.append(f"Min Facility Score: {entities['min_facility_score']}")
    if entities.get("min_reviews"):
        output.append(f"Min Reviews: {entities['min_reviews']}")
    if entities.get("limit"):
        output.append(f"Limit: {entities['limit']}")
    return "\n".join(output) if output else "❌ No entities detected"

if __name__ == "__main__":
    extractor = EnhancedEntityExtractor()
    test_queries = [
        "Looking for hotels in Rome with cleanliness 9, comfort 8.5, and facilities 8",
        "Find hotels with in Mexico cleanliness rating 9, comfort score 8.5, and facilities level 8",
        "Looking for hotels for family with children aged 5, 8, and 10 years old",
        "Hotels for couple aged 25-30 with cleanliness above 8 and comfort 9",
        "Hotels for people in their 30s with staff rating 9",
        "9 cleanliness, 8.5 comfort, and 8 facilities required",
        "Family with kids 5, 8, 10 looking for hotels",
        "cleanliness: 9, comfort: 8.5, facilities: 8",
        "Business hotel for man aged 35 with cleanliness 9+ and staff minimum 8",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i:02d}: {query}")
        print(f"{'='*80}")
        entities = extractor.extract_all(query)
        formatted = format_entities_output(entities)
        print(f"\n{formatted}")