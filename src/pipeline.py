import sys
from pathlib import Path
from typing import Dict, List, Any
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
# from preprocessing import hybrid_intent_detection, extract_entities, clean_text
from baseline_retreiver import BaselineRetriever


class HotelSearchPipeline:
    
    def __init__(self, config_path: str = None, queries_path: str = None):
        self.retriever = BaselineRetriever(
            config_path=config_path,
            queries_path=queries_path
        )
        
        self.intent_mapping = {
            "hotel_search": "hotel_search",
            "hotel_details": "hotel_search",
            "hotel_reviews": "review_lookup",
            "hotel_recommendation": "recommendation",
            "amenity_filtering": "facility_search",
            "location_query": "hotel_search",
            "visa_requirements": "visa_query",
            "general_question": "hotel_search",
            "unknown": "hotel_search"
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"Processing Query: {query}")
        print(f"{'='*80}")
        
        cleaned = clean_text(query)
        print(f"✓ Cleaned: {cleaned}")
        
        intent_result = hybrid_intent_detection(query)
        intent = intent_result.get("intent", "unknown")
        intent_reason = intent_result.get("reason", "No reason")
        intent_method = intent_result.get("method", "unknown")
        print(f"✓ Intent: {intent} (method: {intent_method})")
        print(f"  Reason: {intent_reason}")
        
        entities = extract_entities(query)
        print(f"✓ Entities: {entities}")
        
        retriever_intent, retriever_params = self._map_to_retriever(
            intent, entities, query
        )
        print(f"✓ Retriever Intent: {retriever_intent}")
        print(f"✓ Retriever Params: {retriever_params}")
        
        try:
            results = self.retriever.retrieve(retriever_intent, retriever_params)
            print(f"✓ Retrieved {len(results)} results")
        except Exception as e:
            print(f"✗ Retrieval Error: {e}")
            results = []
        
        return {
            "query": query,
            "cleaned_query": cleaned,
            "intent": intent,
            "intent_reason": intent_reason,
            "intent_method": intent_method,
            "entities": entities,
            "retriever_intent": retriever_intent,
            "retriever_params": retriever_params,
            "results": results,
            "results_count": len(results)
        }
    
    def _map_to_retriever(self, intent: str, entities: Dict, query: str) -> tuple:
        retriever_intent = self.intent_mapping.get(intent, "hotel_search")
        
        params = {"limit": 10}
        
        if entities.get("cities"):
            params["city"] = entities["cities"][0].title()
        
        if entities.get("hotels"):
            params["hotel_name"] = entities["hotels"][0]
        
        if entities.get("countries"):
            params["country"] = entities["countries"][0].title()
        
        if entities.get("traveler_types"):
            params["type"] = entities["traveler_types"][0]
        
        if intent == "visa_requirements":
            retriever_intent = "visa_query"
            countries = entities.get("countries", [])
            if len(countries) >= 2:
                params["country_from"] = countries[0].title()
                params["country_to"] = countries[1].title()
            else:
                query_lower = query.lower()
                if " to " in query_lower:
                    parts = query_lower.split(" to ")
                    if len(parts) >= 2:
                        from_part = parts[0].strip().split()[-1]
                        to_part = parts[1].strip().split()[0]
                        params["country_from"] = from_part.title()
                        params["country_to"] = to_part.title()
        
        if intent == "hotel_reviews":
            retriever_intent = "review_lookup"
            if entities.get("hotels"):
                params["hotel_name"] = entities["hotels"][0]
        
        query_lower = query.lower()
        if "rating" in query_lower or "star" in query_lower:
            import re
            numbers = re.findall(r'\b([1-5])\s*(?:star|rating)', query_lower)
            if numbers:
                params["min_rating"] = float(numbers[0])
        
        if intent == "amenity_filtering":
            retriever_intent = "facility_search"
            facilities = ["pool", "wifi", "parking", "gym", "spa", "breakfast"]
            for facility in facilities:
                if facility in query_lower:
                    params["facility"] = facility
                    break
        
        return retriever_intent, params
    
    def display_results(self, result: Dict[str, Any], max_display: int = 5):
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        
        results = result["results"]
        if not results:
            print("No results found.")
            return
        
        print(f"Found {len(results)} result(s). Showing top {min(len(results), max_display)}:\n")
        
        for i, r in enumerate(results[:max_display], 1):
            print(f"{i}. {self._format_result(r)}")
            print()
    
    def _format_result(self, result: Dict) -> str:
        if "hotel_id" in result and "name" in result:
            parts = [f"Hotel: {result['name']}"]
            if "star_rating" in result and result["star_rating"]:
                parts.append(f"Rating: {result['star_rating']}★")
            if "cleanliness" in result and result["cleanliness"]:
                parts.append(f"Cleanliness: {result['cleanliness']}")
            if "reviews" in result:
                parts.append(f"Reviews: {result['reviews']}")
            if "avg_facilities" in result:
                parts.append(f"Facilities: {result['avg_facilities']:.2f}")
            return " | ".join(parts)
        
        elif "review_id" in result:
            text = result.get("review_text", "")[:100]
            score = result.get("score_overall", "N/A")
            return f"Review (Score: {score}): {text}..."
        
        elif "visa_type" in result:
            return f"Visa Type: {result['visa_type']}"
        
        else:
            return str(result)
    
    def close(self):
        self.retriever.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hotel Search Pipeline")
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to Neo4j config file')
    parser.add_argument('--queries', type=str, default=None,
                       help='Path to Cypher queries file')
    parser.add_argument('--query', type=str, default=None,
                       help='Single query to process')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo queries')
    
    args = parser.parse_args()
    
    pipeline = HotelSearchPipeline(
        config_path=args.config,
        queries_path=args.queries
    )
    
    try:
        if args.query:
            # Process single query
            result = pipeline.process_query(args.query)
            pipeline.display_results(result)
        
        elif args.demo:
            demo_queries = [
                "Find me hotels in Cairo",
                "Show me 5-star hotels in Dubai",
                "What are the reviews for The Royal Compass?",
                "Do Egyptians need a visa to travel to Thailand?",
                "I need a hotel with a swimming pool",
                "Which hotels are best for business travelers?",
                "Show me the top rated hotels in Paris",
                "Find hotels in Egypt with at least 100 reviews",
            ]
            
            for query in demo_queries:
                result = pipeline.process_query(query)
                pipeline.display_results(result, max_display=3)
                print("\n")
        
        else:
            print("Hotel Search Pipeline - Interactive Mode")
            print("Type 'quit' or 'exit' to stop\n")
            
            while True:
                query = input("\nEnter your query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                result = pipeline.process_query(query)
                pipeline.display_results(result)
    
    finally:
        pipeline.close()
        print("\nPipeline closed.")


if __name__ == "__main__":
    main()