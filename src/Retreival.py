import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_retreiver import BaselineRetriever
from Preprocessing.entities_extraction import extract_entities, format_entities_output
from Preprocessing.intent_classifier import hybrid_intent_detection
from embeddings_retreiver import EmbeddingRetriever, load_config


class HybridHotelSearchPipeline:
    def __init__(self, config_path: str = None, queries_path: str = None):
        print("Initializing Hybrid Hotel Search Pipeline...")
        
        if config_path is None:
            config_path = "data/config.txt"
        
        uri, user, password = load_config(config_path)
        
        try:
            self.baseline_retriever = BaselineRetriever(
                config_path=config_path,
                queries_path=queries_path
            )
            print("Baseline Retriever initialized")
        except Exception as e:
            print(f"Failed to initialize baseline retriever: {e}")
            raise
        
        try:
            self.embedding_retriever_minilm = EmbeddingRetriever(uri, user, password, "minilm")
            print("MiniLM Embedding Retriever initialized")
        except Exception as e:
            print(f"Failed to initialize MiniLM retriever: {e}")
            raise
        
        try:
            self.embedding_retriever_mpnet = EmbeddingRetriever(uri, user, password, "mpnet")
            print("MPNet Embedding Retriever initialized")
        except Exception as e:
            print(f"Failed to initialize MPNet retriever: {e}")
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
        if hasattr(self, 'baseline_retriever'):
            self.baseline_retriever.close()
        if hasattr(self, 'embedding_retriever_minilm'):
            self.embedding_retriever_minilm.close()
        if hasattr(self, 'embedding_retriever_mpnet'):
            self.embedding_retriever_mpnet.close()
    
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
            "indian": "India", "indians": "India",
            "egyptian": "Egypt", "egyptians": "Egypt",
            "american": "United States", "americans": "United States",
            "british": "United Kingdom",
            "french": "France", "german": "Germany",
            "chinese": "China", "japanese": "Japan",
            "italian": "Italy", "spanish": "Spain",
            "canadian": "Canada", "australian": "Australia",
            "brazilian": "Brazil", "mexican": "Mexico",
            "russian": "Russia", "saudi": "Saudi Arabia",
            "emirati": "United Arab Emirates",
            "turkish": "Turkey", "thai": "Thailand",
            "korean": "South Korea", "vietnamese": "Vietnam",
            "indonesian": "Indonesia", "malaysian": "Malaysia",
            "singaporean": "Singapore", "filipino": "Philippines",
            "pakistani": "Pakistan", "bangladeshi": "Bangladesh",
            "nigerian": "Nigeria", "south african": "South Africa",
            "kenyan": "Kenya", "moroccan": "Morocco",
            "algerian": "Algeria", "tunisian": "Tunisia",
            "jordanian": "Jordan", "lebanese": "Lebanon",
            "iraqi": "Iraq", "iranian": "Iran",
            "israeli": "Israel", "greek": "Greece",
            "portuguese": "Portugal", "dutch": "Netherlands",
            "belgian": "Belgium", "swiss": "Switzerland",
            "austrian": "Austria", "swedish": "Sweden",
            "norwegian": "Norway", "danish": "Denmark",
            "finnish": "Finland", "polish": "Poland",
            "ukrainian": "Ukraine", "czech": "Czech Republic",
            "hungarian": "Hungary", "romanian": "Romania",
            "argentine": "Argentina", "argentinian": "Argentina",
            "chilean": "Chile", "colombian": "Colombia",
            "peruvian": "Peru", "venezuelan": "Venezuela",
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
    
    def retrieve_baseline_results(self, intent: str, params: Dict[str, Any]) -> List[Dict]:
        print("\nStep 4A: Querying Database (Baseline)...")
        
        retriever_intent = self.normalize_intent(intent)
        print(f"   Using retriever intent: {retriever_intent}")
        
        try:
            results = self.baseline_retriever.retrieve(retriever_intent, params)
            print(f"   Retrieved {len(results)} baseline results")
            return results
        except Exception as e:
            print(f"   Query failed: {e}")
            return []
    
    def retrieve_embedding_results(self, query: str, entities: Dict[str, Any], 
                                   limit: int = 5) -> Dict[str, List[Dict]]:
        print("\nStep 4B: Querying with Embeddings...")
        
        city_filter = entities.get("cities", [None])[0]
        if city_filter:
            city_filter = city_filter.title()
        
        results = {}
        
        print(f"   Searching with MiniLM...")
        try:
            minilm_results = self.embedding_retriever_minilm.search(
                query, 
                limit=limit,
                city_filter=city_filter,
                auto_detect_city=True
            )
            results["minilm"] = minilm_results
            print(f"   Retrieved {len(minilm_results)} MiniLM results")
        except Exception as e:
            print(f"   MiniLM search failed: {e}")
            results["minilm"] = []
        
        print(f"   Searching with MPNet...")
        try:
            mpnet_results = self.embedding_retriever_mpnet.search(
                query,
                limit=limit,
                city_filter=city_filter,
                auto_detect_city=True
            )
            results["mpnet"] = mpnet_results
            print(f"   Retrieved {len(mpnet_results)} MPNet results")
        except Exception as e:
            print(f"   MPNet search failed: {e}")
            results["mpnet"] = []
        
        return results
    
    def fetch_hotel_reviews(self, hotel_id: str, limit: int = 5) -> List[Dict]:
        query = """
        MATCH (h:Hotel {hotel_id:$hotel_id})<-[:REVIEWED]-(r:Review)
        RETURN r.review_id AS review_id,
            r.review_text AS review_text,
            r.review_date AS review_date,
            r.score_overall AS score_overall
        ORDER BY r.review_date DESC 
        LIMIT $limit
        """

        try:
            with self.baseline_retriever.driver.session() as session:
                result = session.run(query, {"hotel_id": hotel_id, "limit": limit})
                return [record.data() for record in result]
        except Exception as e:
            print(f"   Failed to fetch reviews for {hotel_id}: {e}")
            return []
    
    def format_baseline_results(self, results: List[Dict], intent: str) -> Dict[str, Any]:
        if intent == "visa_requirements":
            return {
                "visa_info": results[0] if results else None,
                "nodes": [],
                "reviews": []
            }
        elif intent in ["hotel_reviews", "review_lookup"]:
            return {
                "nodes": [],
                "reviews": results
            }
        else:
            formatted_nodes = []
            for node in results:
                formatted_node = {
                    "id": node.get("hotel_id"),
                    "name": node.get("name"),
                    "city": node.get("city"),
                    "country": node.get("country"),
                    "rating": node.get("star_rating"),
                    "cleanliness": node.get("cleanliness"),
                }
                
                if node.get("hotel_id"):
                    formatted_node["reviews"] = self.fetch_hotel_reviews(node["hotel_id"], limit=3)
                else:
                    formatted_node["reviews"] = []
                
                formatted_nodes.append(formatted_node)
            
            return {
                "nodes": formatted_nodes,
                "reviews": []
            }
    
    def format_embedding_results(self, embedding_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        formatted = {}
        
        for model_name, results in embedding_results.items():
            formatted_nodes = []
            for node in results:
                formatted_node = {
                    "id": node.get("hotel_id"),
                    "name": node.get("name"),
                    "city": node.get("city"),
                    "country": node.get("country"),
                    "rating": node.get("star_rating"),
                    "cleanliness": node.get("cleanliness_base"),
                    "similarity_score": node.get("score"),
                }
                
                if node.get("hotel_id"):
                    formatted_node["reviews"] = self.fetch_hotel_reviews(node["hotel_id"], limit=3)
                else:
                    formatted_node["reviews"] = []
                
                formatted_nodes.append(formatted_node)
            
            formatted[model_name] = {
                "similar_nodes": formatted_nodes,
                "reviews": []
            }
        
        return formatted
    
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
                "embedding_results": {
                    "minilm": {"similar_nodes": [], "reviews": []},
                    "mpnet": {"similar_nodes": [], "reviews": []}
                },
                "error": "Could not determine query intent"
            }
        
        entities = self.extract_entities_from_query(query)
        params = self.map_entities_to_params(intent, entities, query)
        
        baseline_results = self.retrieve_baseline_results(intent, params)
        
        embedding_results = {}
        if intent not in ["visa_requirements"]:
            embedding_results = self.retrieve_embedding_results(query, entities, limit=5)
        
        baseline_formatted = self.format_baseline_results(baseline_results, intent)
        embedding_formatted = self.format_embedding_results(embedding_results)
        
        output = {
            "query": query,
            "baseline_results": baseline_formatted,
            "embedding_results": embedding_formatted
        }
        
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"Baseline: {len(baseline_formatted.get('nodes', []))} nodes, {len(baseline_formatted.get('reviews', []))} reviews")
        if embedding_formatted:
            for model_name, model_results in embedding_formatted.items():
                print(f"{model_name.upper()}: {len(model_results.get('similar_nodes', []))} nodes")
        print(f"\n{json.dumps(output, indent=2)}")
        
        return output


def interactive_mode(config_path=None, queries_path=None):
    print("\n" + "="*80)
    print("HYBRID HOTEL SEARCH PIPELINE - INTERACTIVE MODE")
    print("="*80)
    print("\nType your queries below. Type 'exit' or 'quit' to stop.\n")
    
    pipeline = HybridHotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
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
    print("HYBRID HOTEL SEARCH PIPELINE - BATCH MODE")
    print("="*80)
    
    pipeline = HybridHotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
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
        description="Hybrid Hotel Search Pipeline - Baseline + Embeddings"
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
            "Do Egyptians need a visa to travel to France?",
            "Find hotels in Paris for a family with children aged 5-10",
            "What are the best business hotels in London?",
            "Show me hotels with good facilities in Tokyo",
            "I want a hotel for a solo female traveler in their 30s",
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