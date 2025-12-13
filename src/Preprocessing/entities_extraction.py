"""
Enhanced Entity Extraction Module
Combines spaCy NER with custom facility rating extraction
FIXED: Properly distinguishes between facility ratings and ages
USES: hotels.csv columns for cities and countries instead of spaCy
"""

import pandas as pd
from pathlib import Path
import spacy
import re
from typing import Dict, List, Optional, Tuple

# ==================== CONFIGURATION ====================

class EntityExtractorConfig:
    """Configuration for entity extraction"""
    
    # Data paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  
    DATA_PATH = BASE_DIR / "data" / "hotels.csv"
    
    # Predefined categories
    TRAVELER_TYPES = ["family", "couple", "solo", "business", "group", "friends"]
    
    FACILITIES = [
        "cleanliness", 
        "comfort", 
        "facilities", 
        "location", 
        "staff", 
        "value for money",
        "value"  # Short form
    ]
    
    # Gender patterns with word boundaries
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


# ==================== ENHANCED ENTITY EXTRACTOR ====================

class EnhancedEntityExtractor:
    """
    Enhanced entity extractor using spaCy + custom patterns
    Extracts: hotels, cities, countries, traveler types, facilities,
             facility ratings, gender, age, nationalities
    USES CSV COLUMNS FOR CITIES AND COUNTRIES INSTEAD OF SPACY
    """
    
    def __init__(self, data_path: Optional[str] = None, spacy_model: str = "en_core_web_sm"):
        """
        Initialize entity extractor
        
        Args:
            data_path: Path to hotels.csv (optional)
            spacy_model: SpaCy model to use
        """
        self.config = EntityExtractorConfig()
        
        # Load hotels, cities, and countries from CSV
        self._load_hotel_data(data_path)
        
        # Load spaCy model (only for nationalities now)
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"✓ Loaded spaCy model: {spacy_model}")
        except OSError:
            print(f"✗ SpaCy model '{spacy_model}' not found. Install with:")
            print(f"  python -m spacy download {spacy_model}")
            self.nlp = None
    
    def _load_hotel_data(self, data_path: Optional[str] = None):
        """Load hotel names, cities, and countries from CSV"""
        try:
            path = Path(data_path) if data_path else self.config.DATA_PATH
            hotels_df = pd.read_csv(path)
            
            # Load hotels
            self.hotels = hotels_df['hotel_name'].str.lower().tolist()
            
            # Load cities and countries from CSV columns
            self.cities = hotels_df['city'].str.lower().dropna().unique().tolist()
            self.countries = hotels_df['country'].str.lower().dropna().unique().tolist()
            
            print(f"✓ Loaded {len(self.hotels)} hotels, {len(self.cities)} cities, {len(self.countries)} countries from CSV")
        except FileNotFoundError:
            print(f"✗ Hotels CSV not found at {path}")
            self.hotels = []
            self.cities = []
            self.countries = []
        except KeyError as e:
            print(f"✗ Missing column in CSV: {e}")
            self.hotels = []
            self.cities = []
            self.countries = []
    
    def extract_all(self, query: str) -> Dict[str, any]:
        """
        Extract all entities from query
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with all extracted entities
        """
        query_lower = query.lower()
        
        entities = {
            "hotels": self._extract_hotels(query_lower),
            "cities": self._extract_cities_from_csv(query_lower),  # CSV-based
            "countries": self._extract_countries_from_csv(query_lower),  # CSV-based
            "traveler_types": self._extract_traveler_types(query_lower),
            "facilities": self._extract_facilities(query_lower),
            "facility_ratings": self._extract_facility_ratings(query),
            "nationality": [],
            "age_numbers": self._extract_age_numbers(query_lower),
            "gender": self._extract_gender(query_lower),
            "star_rating": self._extract_star_rating(query_lower),
            "min_rating": self._extract_min_rating(query_lower),
        }
        
        # Use spaCy ONLY for nationalities (not cities/countries anymore)
        if self.nlp:
            spacy_entities = self._extract_spacy_entities(query)
            entities["nationality"].extend(spacy_entities["nationality"])
        
        # Remove duplicates
        for key in ["hotels", "cities", "countries", "traveler_types", 
                   "facilities", "nationality", "gender"]:
            entities[key] = list(set(entities[key]))
        
        # Sort age numbers
        entities["age_numbers"] = sorted(entities["age_numbers"])
        
        return entities
    
    # ==================== HOTEL EXTRACTION ====================
    
    def _extract_hotels(self, query: str) -> List[str]:
        """Extract hotel names from loaded CSV data"""
        found_hotels = []
        
        # Direct matching
        for hotel in self.hotels:
            if hotel in query:
                found_hotels.append(hotel)
        
        # Check for partial matches (word-by-word)
        query_words = query.split()
        for hotel in self.hotels:
            hotel_words = hotel.split()
            # If 2+ consecutive words match
            for i in range(len(query_words) - len(hotel_words) + 1):
                if ' '.join(query_words[i:i+len(hotel_words)]) == hotel:
                    found_hotels.append(hotel)
        
        return list(set(found_hotels))
    
    # ==================== CITY EXTRACTION FROM CSV ====================
    
    def _extract_cities_from_csv(self, query: str) -> List[str]:
        """Extract cities from CSV column data"""
        found_cities = []
        
        if not hasattr(self, 'cities') or not self.cities:
            return found_cities
        
        for city in self.cities:
            if city in query:
                found_cities.append(city)
        
        return list(set(found_cities))
    
    # ==================== COUNTRY EXTRACTION FROM CSV ====================
    
    def _extract_countries_from_csv(self, query: str) -> List[str]:
        """Extract countries from CSV column data"""
        found_countries = []
        
        if not hasattr(self, 'countries') or not self.countries:
            return found_countries
        
        for country in self.countries:
            if country in query:
                found_countries.append(country)
        
        return list(set(found_countries))
    
    # ==================== TRAVELER TYPE EXTRACTION ====================
    
    def _extract_traveler_types(self, query: str) -> List[str]:
        """Extract traveler types"""
        found_types = []
        
        for ttype in self.config.TRAVELER_TYPES:
            if ttype in query:
                found_types.append(ttype)
        
        return found_types
    
    # ==================== FACILITIES EXTRACTION ====================
    
    def _extract_facilities(self, query: str) -> List[str]:
        """Extract facility mentions"""
        found_facilities = []
        
        for facility in self.config.FACILITIES:
            if facility in query:
                # Normalize "value for money" to "value"
                normalized = "value" if "value" in facility else facility
                if normalized not in found_facilities:
                    found_facilities.append(normalized)
        
        return found_facilities
    
    # ==================== FIXED FACILITY RATINGS EXTRACTION ====================
    
    def _extract_facility_ratings(self, query: str) -> Dict[str, float]:
        """
        Extract specific facility ratings from query
        FIXED: Better pattern matching to avoid confusion with age numbers
        
        Examples:
        - "cleanliness rating 9" -> {'cleanliness': 9.0}
        - "comfort above 8.5" -> {'comfort': 8.5}
        - "staff 9+ and value for money 8" -> {'staff': 9.0, 'value': 8.0}
        
        Returns:
            Dict mapping facility name to minimum rating
        """
        facility_ratings = {}
        query_lower = query.lower()
        
        # Define facilities to look for
        facilities = ['cleanliness', 'comfort', 'facilities', 'location', 'staff', 'value']
        
        # IMPORTANT: First, let's extract age numbers to avoid confusion
        age_numbers = self._extract_age_numbers(query_lower)
        
        # More specific patterns for facility ratings
        patterns = [
            # Pattern 1: "cleanliness rating 9", "comfort score 8.5" (specific keywords)
            (r'\b({facility})\s+(?:rating|score|level)\s+(?:of\s+)?(?:at least\s+)?(\d+(?:\.\d+)?)\b', 'direct_with_keyword'),
            
            # Pattern 2: "cleanliness above 9", "comfort over 8" (comparison words)
            (r'\b({facility})\s+(?:above|over|greater than|more than|at least|minimum|>=|>)\s*(\d+(?:\.\d+)?)\b', 'comparison'),
            
            # Pattern 3: "cleanliness 9+", "comfort 8.5+" (plus sign)
            (r'\b({facility})\s+(\d+(?:\.\d+)?)\+\b', 'with_plus'),
            
            # Pattern 4: "rating 9 for cleanliness", "score 8.5 for comfort" (rating before)
            (r'\b(?:rating|score)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s+(?:for|on|in)\s+({facility})\b', 'rating_first'),
            
            # Pattern 5: "with cleanliness 9", "has comfort 8.5" (with preposition)
            (r'\b(?:with|has|having)\s+({facility})\s+(?:of\s+)?(\d+(?:\.\d+)?)\b', 'with_preposition'),
            
            # Pattern 6: "cleanliness: 9", "comfort: 8.5" (colon)
            (r'\b({facility})\s*:\s*(\d+(?:\.\d+)?)\b', 'with_colon'),
            
            # Pattern 7: Specific format like "cleanliness=9", "comfort=8.5"
            (r'\b({facility})\s*=\s*(\d+(?:\.\d+)?)\b', 'with_equals'),
        ]
        
        for facility in facilities:
            for pattern_template, pattern_type in patterns:
                # Replace {facility} placeholder
                pattern = pattern_template.replace('{facility}', facility)
                
                # Also check for "value for money" variant
                if facility == 'value':
                    pattern_variants = [
                        pattern,
                        pattern.replace('value', 'value for money')
                    ]
                else:
                    pattern_variants = [pattern]
                
                for p in pattern_variants:
                    matches = re.finditer(p, query_lower)
                    for match in matches:
                        groups = match.groups()
                        
                        # Determine which group is the rating number
                        rating = None
                        if pattern_type in ['direct_with_keyword', 'comparison', 'with_plus', 'with_preposition', 'with_colon', 'with_equals']:
                            # Format: facility rating
                            if groups[1] and groups[1].replace('.', '').isdigit():
                                rating = float(groups[1])
                        elif pattern_type == 'rating_first':
                            # Format: rating facility
                            if groups[0] and groups[0].replace('.', '').isdigit():
                                rating = float(groups[0])
                        
                        if rating is not None:
                            # Check if this might be an age number (to avoid confusion)
                            is_likely_age = False
                            
                            # Heuristic 1: If rating is in age numbers list
                            if int(rating) in age_numbers:
                                # Heuristic 2: Check context around the match
                                context_start = max(0, match.start() - 10)
                                context_end = min(len(query_lower), match.end() + 10)
                                context = query_lower[context_start:context_end]
                                
                                # Age-related keywords near the match
                                age_keywords = ['age', 'aged', 'years', 'old', 'year old', 'yr']
                                facility_keywords = ['rating', 'score', 'level', 'above', 'over', 'minimum']
                                
                                age_keyword_nearby = any(keyword in context for keyword in age_keywords)
                                facility_keyword_nearby = any(keyword in context for keyword in facility_keywords)
                                
                                # If it has age keywords but not facility keywords, it's likely age
                                if age_keyword_nearby and not facility_keyword_nearby:
                                    is_likely_age = True
                            
                            # Heuristic 3: Rating values typical for facility scores (usually 0-10)
                            # Age numbers are usually integers 1-99, facility ratings often have decimals
                            if rating.is_integer() and 1 <= rating <= 100 and '.' not in str(rating):
                                # Could be age or rating, need more context
                                pass
                            
                            if not is_likely_age and 0 <= rating <= 10:  # Valid rating range
                                # Normalize facility name
                                norm_facility = 'value' if facility == 'value for money' else facility
                                facility_ratings[norm_facility] = rating
        
        # Additional check: Look for patterns like "9 cleanliness, 8.5 comfort, 8 facilities"
        # This pattern was causing the confusion
        list_pattern = r'\b(\d+(?:\.\d+)?)\s*,\s*(cleanliness|comfort|facilities|location|staff|value(?: for money)?)\b'
        list_matches = re.findall(list_pattern, query_lower)
        
        for rating_str, facility in list_matches:
            if rating_str.replace('.', '').isdigit():
                rating = float(rating_str)
                # Check if this could be an age number
                if int(rating) not in age_numbers and 0 <= rating <= 10:
                    norm_facility = 'value' if 'value' in facility else facility
                    facility_ratings[norm_facility] = rating
        
        return facility_ratings
    
    # ==================== GENDER EXTRACTION ====================
    
    def _extract_gender(self, query: str) -> List[str]:
        """Extract gender entities using word boundaries"""
        genders_found = []
        
        # Check for female patterns first (to avoid "man" in "woman")
        for gender, patterns in self.config.GENDERS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    if gender not in genders_found:
                        genders_found.append(gender)
        
        return genders_found
    
    # ==================== AGE EXTRACTION ====================
    
    def _extract_age_numbers(self, query: str) -> List[int]:
        """
        Extract age numbers from query
        FIXED: Better pattern matching to avoid confusion with facility ratings
        
        Examples:
        - "aged 25" -> [25]
        - "5-10 years" -> [5, 10]
        - "in their 30s" -> [30]
        - "25+" -> [25]
        """
        age_numbers = []
        
        # Age-related keywords that indicate we're talking about age
        age_keywords = ['age', 'aged', 'years', 'old', 'year old', 'yr', 'yo', 'y.o.', 'years old']
        
        # Look for age context first
        has_age_context = any(keyword in query for keyword in age_keywords)
        
        # Only extract ages if we have age context OR specific age patterns
        if has_age_context:
            # Exact ages with context: "25 years old", "aged 35"
            exact_patterns = [
                r'(\d+)\s*years?\s*old\b',
                r'\baged\s+(\d+)\b',
                r'\bage\s+(\d+)\b',
                r'\b(\d+)\s*years?\b',  # Only if we have age context
            ]
            
            # Age ranges with context: "5-10 years", "18 to 25"
            range_patterns = [
                r'\b(\d+)\s*-\s*(\d+)\s*years?\b',
                r'\b(\d+)\s+to\s+(\d+)\s*years?\b',
                r'\b(\d+)\s+and\s+(\d+)\s*years?\b',
            ]
            
            # Decades with context: "in their 30s", "40s"
            decade_pattern = r'\b(?:in\s+their\s+)?(\d+)s\b'
            
            # Plus patterns with context: "25+", "65+"
            plus_pattern = r'\b(\d+)\+\b'
        else:
            # Without age context, be more conservative
            exact_patterns = [
                r'\baged\s+(\d+)\b',
                r'\bage\s+(\d+)\b',
                r'\b(\d+)\s*years?\s*old\b',
            ]
            
            range_patterns = [
                r'\b(\d+)\s*-\s*(\d+)\s*years?\b',
            ]
            
            decade_pattern = r'\b(?:in\s+their\s+)?(\d+)s\b'
            plus_pattern = r'\b(\d+)\+\b'
        
        # Extract ranges first
        for pattern in range_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                for group in match.groups():
                    if group and group.isdigit():
                        age = int(group)
                        if 0 < age < 120:  # Valid age range
                            age_numbers.append(age)
        
        # Extract exact ages
        for pattern in exact_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                if match.group(1) and match.group(1).isdigit():
                    age = int(match.group(1))
                    if 0 < age < 120 and age not in age_numbers:
                        age_numbers.append(age)
        
        # Extract decades
        decade_matches = re.finditer(decade_pattern, query)
        for match in decade_matches:
            if match.group(1) and match.group(1).isdigit():
                decade = int(match.group(1))
                if 10 <= decade <= 90 and decade not in age_numbers:
                    age_numbers.append(decade)
        
        # Extract plus patterns
        plus_matches = re.finditer(plus_pattern, query)
        for match in plus_matches:
            if match.group(1) and match.group(1).isdigit():
                age = int(match.group(1))
                if 0 < age < 120 and age not in age_numbers:
                    age_numbers.append(age)
        
        return sorted(list(set(age_numbers)))
    
    # ==================== STAR RATING EXTRACTION ====================
    
    def _extract_star_rating(self, query: str) -> Optional[int]:
        """Extract star rating"""
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
    
    # ==================== MINIMUM RATING EXTRACTION ====================
    
    def _extract_min_rating(self, query: str) -> Optional[float]:
        """Extract minimum overall rating requirement"""
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
    
    # ==================== SPACY NER EXTRACTION (NATIONALITIES ONLY) ====================
    
    def _extract_spacy_entities(self, query: str) -> Dict[str, List[str]]:
        """Use spaCy ONLY for nationality extraction """
        entities = {
            "nationality": []
        }
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(query)
        
        for ent in doc.ents:
            if ent.label_ == "NORP":  # Nationalities
                entities["nationality"].append(ent.text.lower())
            # REMOVED: GPE and LOC handling since we use CSV data
        
        return entities
    
    # ==================== HELPER: DIFFERENTIATE AGE VS RATING ====================
    
    def _is_likely_age(self, number: float, context: str) -> bool:
        """
        Determine if a number is more likely to be an age or a rating
        
        Args:
            number: The number to check
            context: Text around the number
            
        Returns:
            True if likely age, False if likely rating
        """
        # Age indicators
        age_keywords = ['age', 'aged', 'years', 'old', 'year old', 'yr', 'yo', 'y.o.']
        age_indicators = any(keyword in context.lower() for keyword in age_keywords)
        
        # Rating indicators
        rating_keywords = ['rating', 'score', 'level', 'above', 'over', 'minimum', 'stars']
        rating_indicators = any(keyword in context.lower() for keyword in rating_keywords)
        
        # Value ranges
        # Ages are typically 0-120, ratings typically 0-10 or 0-5
        is_age_range = 0 <= number <= 120
        is_rating_range = 0 <= number <= 10
        
        # Decimal check: ratings often have decimals (8.5), ages rarely do
        has_decimal = '.' in str(number)
        
        # Heuristic decision
        if age_indicators and not rating_indicators:
            return True
        elif rating_indicators and not age_indicators:
            return False
        elif has_decimal and is_rating_range:
            return False  # Decimal numbers in 0-10 range are likely ratings
        elif is_age_range and number.is_integer() and not has_decimal:
            return True  # Integer in age range without rating context
        else:
            return False  # Default to rating if unsure


# ==================== OUTPUT FORMATTING ====================

def format_entities_output(entities: Dict) -> str:
    """Create formatted string of extracted entities"""
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
    
    return "\n".join(output) if output else "❌ No entities detected"


# ==================== MAIN / TESTING ====================

if __name__ == "__main__":
    print("="*100)
    print("ENHANCED ENTITY EXTRACTION - CSV BASED CITIES/COUNTRIES")
    print("Properly distinguishes between facility ratings and ages")
    print("Uses hotels.csv columns for cities and countries")
    print("="*100)
    
    # Initialize extractor
    extractor = EnhancedEntityExtractor()
    
    # Test queries specifically designed to test the confusion cases
    test_queries = [
        # Problematic query from before
        "Looking for hotels in Rome with cleanliness 9, comfort 8.5, and facilities 8",
        
        # Clear facility ratings (should extract all three)
        "Find hotels with in Mexico cleanliness rating 9, comfort score 8.5, and facilities level 8",
        
        # Clear age query (should NOT extract as facility ratings)
        "Looking for hotels for family with children aged 5, 8, and 10 years old",
        
        # Mixed query (should differentiate)
        "Hotels for couple aged 25-30 with cleanliness above 8 and comfort 9",
        
        # Edge case: numbers that could be both
        "Hotels for people in their 30s with staff rating 9",
        
        # List format with commas (the problematic pattern)
        "9 cleanliness, 8.5 comfort, and 8 facilities required",
        
        # Age query that looks like facility ratings
        "Family with kids 5, 8, 10 looking for hotels",
        
        # Facility ratings without context clues
        "cleanliness: 9, comfort: 8.5, facilities: 8",
        
        # More complex mixed example
        "Business hotel for man aged 35 with cleanliness 9+ and staff minimum 8",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i:02d}: {query}")
        print(f"{'='*80}")
        
        entities = extractor.extract_all(query)
        formatted = format_entities_output(entities)
        
        print(f"\n{formatted}")
        
        # Special debugging for confusing cases
        if "9" in query or "8" in query or "10" in query:
            print(f"\n🔍 DEBUG INFO:")
            print(f"  Age numbers extracted: {entities.get('age_numbers', [])}")
            print(f"  Facility ratings: {entities.get('facility_ratings', {})}")
            print(f"  Facilities mentioned: {entities.get('facilities', [])}")
            
            # Check for potential confusion
            for num in entities.get('age_numbers', []):
                if num in [9, 8, 10, 8.5]:
                    print(f"  ⚠️  Note: {num} appears in both age and rating ranges")
    
    print(f"\n{'='*100}")
    print("✓ Testing complete - CSV-based city/country extraction")
    print(f"{'='*100}")