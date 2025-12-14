import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_retreiver import BaselineRetriever
from Preprocessing.entities_extraction import EnhancedEntityExtractor, format_entities_output
from Preprocessing.intent_classifier import hybrid_intent_detection, refine_intent_with_entities

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

        try:
            self.extractor = EnhancedEntityExtractor()
            print("EnhancedEntityExtractor initialized")
        except Exception as e:
            print("Failed to initialize EnhancedEntityExtractor:", e)
            self.extractor = None
    
    def close(self):
        if hasattr(self, 'retriever'):
            self.retriever.close()
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        print("\nStep 1: Classifying Intent...")
        result = hybrid_intent_detection(query)
        intent = result.get("intent", "unknown")
        method = result.get("method", "unknown")
        reason = result.get("reason", "No reason provided")
        print(f"   Initial intent: {intent} (method={method}) - {reason}")
        return result

    def extract_entities_from_query(self, query: str) -> Dict[str, Any]:
        print("\nStep 2: Extracting Entities...")

        if self.extractor:
            entities = self.extractor.extract_all(query)
        else:
            entities = {
                "hotels": [], "cities": [], "countries": [], "traveler_types": [],
                "facilities": [], "facility_ratings": {}, "nationality": [],
                "age_numbers": [], "gender": [], "star_rating": None, "min_rating": None,
                "min_facility_score": None, "min_reviews": None, "limit": 5
            }

        formatted = format_entities_output(entities)
        if not formatted.startswith("No entities detected"):
            print(f"\n{formatted}")
        else:
            print("   No entities detected")
        
        return entities

    def map_entities_to_params(self, intent: str, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        print("\nStep 3: Mapping Entities to Query Parameters...")
        
        if self.extractor:
            params = self.extractor.map_entities_to_params(intent, entities, query)
        else:
            params = {}
            params["limit"] = entities.get("limit", 5)
        
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        if not params or (len(params) == 1 and "limit" in params):
            print("   No specific parameters extracted, using defaults")
        
        return params
    
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
            "traveller_query": "traveller_type_preferences",
            "traveller_type_preferences": "traveller_type_preferences",
            "hotels_with_min_reviews": "hotels_with_min_reviews",
            "hotels_by_traveler_gender_age": "hotels_by_traveler_gender_age",
            "hotels_by_score_range": "hotels_by_score_range",
            "best_value_hotels": "best_value_hotels",
            "hotels_by_location_score": "hotels_by_location_score",
            "hotels_with_best_staff": "hotels_with_best_staff"
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
            return {"visa_info": results[0] if results else None}
        elif intent in ["hotel_reviews", "review_lookup"]:
            return {"nodes": [], "reviews": results}
        else:
            return {"nodes": results, "reviews": []}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"PROCESSING QUERY: {query}")
        print(f"{'='*80}")
        
        intent_result = self.extract_intent(query)
        raw_intent = intent_result.get("intent", "unknown")
        entities = self.extract_entities_from_query(query)
        intent = refine_intent_with_entities(raw_intent, entities, query)
        print(f"\n   Final intent (after refinement): {intent}")
        params = self.map_entities_to_params(intent, entities, query)

        if intent == "hotel_reviews" and not params.get("hotel_id") and params.get("hotel_name"):
            qname = params["hotel_name"]
            try:
                tpl = self.retriever.queries.get("hotel_by_name")
                if tpl:
                    hits = self.retriever.run_query(tpl, {"q": qname, "limit": 1})
                    if hits:
                        params["hotel_id"] = hits[0].get("hotel_id")
                        print(f"   Resolved hotel_name '{qname}' -> hotel_id '{params['hotel_id']}'")
            except Exception as e:
                print(f"   Warning: failed to resolve hotel_name to id: {e}")

        results = self.retrieve_results(intent, params)
        structured_results = self.format_results_structured(results, intent)
        output = {"query": query, "baseline_results": structured_results}
        
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