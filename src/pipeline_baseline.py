import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_retreiver import BaselineRetriever
from Preprocessing.entities_extraction import extract_entities, format_entities_output
from Preprocessing.intent_classifier import hybrid_intent_detection


class HotelSearchPipeline:
    def __init__(self, config_path: str = None, queries_path: str = None):
        print("Initializing Hotel Search Pipeline...")
        
        try:
            self.retriever = BaselineRetriever(
                config_path=config_path,
                queries_path=queries_path
            )
            print("Baseline Retriever initialized successfully")
        except Exception as e:
            print(f"Failed to initialize retriever: {e}")
            raise
        
        self.intent_entity_map = {
            "hotel_search": ["city", "country", "hotel_name", "min_rating"],
            "hotel_details": ["hotel_name", "hotel_id"],
            "hotel_reviews": ["hotel_id", "hotel_name"],
            "hotel_recommendation": ["city", "country", "min_rating"],
            "amenity_filtering": ["facility", "min_facility_score"],
            "location_query": ["city", "country"],
            "visa_requirements": ["country_from", "country_to"],
            "general_question": ["city", "hotel_name"],
        }
    
    def close(self):
        if hasattr(self, 'retriever'):
            self.retriever.close()
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        print("\nStep 1: Classifying Intent...")
        
        query_lower = query.lower()
        visa_keywords = ["visa", "passport", "entry requirement", "travel document", 
                        "need a visa", "require a visa", "visa requirement"]
        
        if any(keyword in query_lower for keyword in visa_keywords):
            result = {
                "intent": "visa_requirements",
                "reason": "Query contains visa-related keywords",
                "method": "rule_based_visa"
            }
        else:
            result = hybrid_intent_detection(query)
        
        intent = result.get("intent", "unknown")
        reason = result.get("reason", "No reason provided")
        method = result.get("method", "unknown")
        
        print(f"   Intent: {intent}")
        print(f"   Method: {method}")
        print(f"   Reason: {reason}")
        
        return result
    
    def extract_entities_from_query(self, query: str) -> Dict[str, Any]:
        print("\nStep 2: Extracting Entities...")
        entities = extract_entities(query)
        
        formatted = format_entities_output(entities)
        if formatted != "No entities detected":
            print(f"\n{formatted}")
        else:
            print("   No entities detected")
        
        return entities
    
    def map_entities_to_params(self, intent: str, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        print("\nStep 3: Mapping Entities to Query Parameters...")
        
        params = {}
        
        if intent == "visa_requirements":
            if entities.get("nationality"):
                nationality = entities["nationality"][0]
                country_from = self.nationality_to_country(nationality)
                params["from"] = country_from
                print(f"   From Country (nationality): {country_from}")
            
            if entities.get("countries"):
                params["to"] = entities["countries"][0].title()
                print(f"   To Country: {params['to']}")
            
            if len(entities.get("countries", [])) >= 2 and not params.get("from"):
                params["from"] = entities["countries"][0].title()
                params["to"] = entities["countries"][1].title()
                print(f"   From Country: {params['from']}")
                print(f"   To Country: {params['to']}")
            
            return params
        
        if entities.get("cities"):
            params["city"] = entities["cities"][0].title()
            print(f"   City: {params['city']}")
        
        if entities.get("countries"):
            params["country"] = entities["countries"][0].title()
            print(f"   Country: {params['country']}")
        
        if entities.get("hotels"):
            params["hotel_name"] = entities["hotels"][0]
            print(f"   Hotel: {params['hotel_name']}")
        
        if entities.get("traveler_types"):
            params["traveler_type"] = entities["traveler_types"][0]
            print(f"   Traveler Type: {params['traveler_type']}")
        
        if entities.get("gender"):
            params["gender"] = entities["gender"][0]
            print(f"   Gender: {params['gender']}")
        
        if entities.get("age_numbers"):
            if len(entities["age_numbers"]) == 1:
                params["age"] = entities["age_numbers"][0]
                print(f"   Age: {params['age']}")
            elif len(entities["age_numbers"]) >= 2:
                params["age_min"] = min(entities["age_numbers"])
                params["age_max"] = max(entities["age_numbers"])
                print(f"   Age Range: {params['age_min']}-{params['age_max']}")
        
        params["limit"] = 5
        
        if not params or params == {"limit": 5}:
            print("   No specific parameters extracted, using defaults")
        
        return params
    
    def nationality_to_country(self, nationality: str) -> str:
        nationality_map = {
            "indian": "India",
            "indians": "India",
            "egyptian": "Egypt",
            "egyptians": "Egypt",
            "american": "United States",
            "americans": "United States",
            "british": "United Kingdom",
            "french": "France",
            "german": "Germany",
            "chinese": "China",
            "japanese": "Japan",
            "italian": "Italy",
            "spanish": "Spain",
            "canadian": "Canada",
            "australian": "Australia",
            "brazilian": "Brazil",
            "mexican": "Mexico",
            "russian": "Russia",
            "saudi": "Saudi Arabia",
            "emirati": "United Arab Emirates",
            "turkish": "Turkey",
            "thai": "Thailand",
            "korean": "South Korea",
            "vietnamese": "Vietnam",
            "indonesian": "Indonesia",
            "malaysian": "Malaysia",
            "singaporean": "Singapore",
            "filipino": "Philippines",
            "pakistani": "Pakistan",
            "bangladeshi": "Bangladesh",
            "nigerian": "Nigeria",
            "south african": "South Africa",
            "kenyan": "Kenya",
            "moroccan": "Morocco",
            "algerian": "Algeria",
            "tunisian": "Tunisia",
            "jordanian": "Jordan",
            "lebanese": "Lebanon",
            "iraqi": "Iraq",
            "iranian": "Iran",
            "israeli": "Israel",
            "greek": "Greece",
            "portuguese": "Portugal",
            "dutch": "Netherlands",
            "belgian": "Belgium",
            "swiss": "Switzerland",
            "austrian": "Austria",
            "swedish": "Sweden",
            "norwegian": "Norway",
            "danish": "Denmark",
            "finnish": "Finland",
            "polish": "Poland",
            "ukrainian": "Ukraine",
            "czech": "Czech Republic",
            "hungarian": "Hungary",
            "romanian": "Romania",
            "argentine": "Argentina",
            "argentinian": "Argentina",
            "chilean": "Chile",
            "colombian": "Colombia",
            "peruvian": "Peru",
            "venezuelan": "Venezuela",
        }
        
        key = nationality.lower().strip()
        return nationality_map.get(key, nationality.title())
    
    def normalize_intent(self, intent: str) -> str:
        intent_mapping = {
            "hotel_search": "recommendation",
            "hotel_details": "hotel_search",
            "hotel_reviews": "review_lookup",
            "hotel_recommendation": "recommendation",
            "amenity_filtering": "facility_search",
            "location_query": "hotel_search",
            "visa_requirements": "visa_query",
            "general_question": "hotel_search",
        }
        
        return intent_mapping.get(intent, "hotel_search")
    
    def retrieve_results(self, intent: str, params: Dict[str, Any]) -> List[Dict]:
        print("\nStep 4: Querying Database...")
        
        retriever_intent = self.normalize_intent(intent)
        print(f"   Using retriever intent: {retriever_intent}")
        
        try:
            results = self.retriever.retrieve(retriever_intent, params)
            print(f"   Retrieved {len(results)} results")
            return results
        except Exception as e:
            print(f"   Query failed: {e}")
            return []
    
    def format_results_structured(self, results: List[Dict], intent: str) -> Dict[str, Any]:
        if intent == "visa_requirements":
            return {
                "visa_info": results[0] if results else None
            }
        elif intent in ["hotel_reviews", "review_lookup"]:
            return {
                "nodes": [],
                "reviews": results
            }
        else:
            return {
                "nodes": results,
                "reviews": []
            }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"PROCESSING QUERY: {query}")
        print(f"{'='*80}")
        
        intent_result = self.extract_intent(query)
        intent = intent_result.get("intent", "unknown")
        
        if intent == "unknown":
            return {
                "query": query,
                "baseline_results": {
                    "nodes": [],
                    "reviews": []
                },
                "error": "Could not determine query intent"
            }
        
        entities = self.extract_entities_from_query(query)
        params = self.map_entities_to_params(intent, entities, query)
        results = self.retrieve_results(intent, params)
        
        structured_results = self.format_results_structured(results, intent)
        
        output = {
            "query": query,
            "baseline_results": structured_results
        }
        
        print(f"\n{'='*80}")
        print(f"RESULTS ({len(results)} found)")
        print(f"{'='*80}")
        print(json.dumps(output, indent=2))
        
        return output


def interactive_mode(config_path=None, queries_path=None):
    print("\n" + "="*80)
    print("HOTEL SEARCH PIPELINE - INTERACTIVE MODE")
    print("="*80)
    print("\nType your queries below. Type 'exit' or 'quit' to stop.\n")
    
    pipeline = HotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
    try:
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                pipeline.process_query(query)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")
                import traceback
                traceback.print_exc()
    
    finally:
        pipeline.close()


def batch_mode(queries: List[str], config_path=None, queries_path=None):
    print("\n" + "="*80)
    print("HOTEL SEARCH PIPELINE - BATCH MODE")
    print("="*80)
    
    pipeline = HotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
    try:
        for i, query in enumerate(queries, 1):
            print(f"\n\n{'#'*80}")
            print(f"# QUERY {i}/{len(queries)}")
            print(f"{'#'*80}")
            
            pipeline.process_query(query)
    
    finally:
        pipeline.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hotel Search Pipeline - End-to-end query processing"
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'batch', 'demo'],
        default='interactive',
        help='Execution mode (default: interactive)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.txt'
    )
    parser.add_argument(
        '--queries',
        type=str,
        default=None,
        help='Path to queries.txt'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_mode(config_path=args.config, queries_path=args.queries)
    
    elif args.mode == 'batch':
        test_queries = [
            "Find me hotels in Cairo",
            "Show me reviews for The Royal Compass",
            "I need a hotel in Dubai with a rating above 4 stars",
            "Find hotels in Paris for a family with children aged 5-10",
            "What are the best business hotels in London?",
            "Show me hotels with good facilities in Tokyo",
        ]
        batch_mode(test_queries, config_path=args.config, queries_path=args.queries)
    
    elif args.mode == 'demo':
        demo_queries = [
            "Find hotels in Cairo",
            "Show reviews for The Royal Compass",
            "Do Egyptians need a visa to France?",
        ]
        batch_mode(demo_queries, config_path=args.config, queries_path=args.queries)


if __name__ == "__main__":
    main()