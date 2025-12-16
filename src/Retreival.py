# import sys
# import os
# from pathlib import Path
# from typing import Dict, Any, List
# import json

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# SRC_DIR = PROJECT_ROOT / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

# from baseline_retreiver import BaselineRetriever
# from Preprocessing.entities_extraction import EnhancedEntityExtractor, format_entities_output
# from Preprocessing.intent_classifier import hybrid_intent_detection, refine_intent_with_entities

# try:
#     from embeddings_retreiver import EmbeddingRetriever
#     EMBEDDINGS_AVAILABLE = True
# except ImportError:
#     EMBEDDINGS_AVAILABLE = False
#     print("Warning: embeddings_retreiver not available, running baseline only")


# class HybridHotelSearchPipeline:
#     def __init__(self, config_path: str = None, queries_path: str = None):
#         print("Initializing Hybrid Hotel Search Pipeline...")
        
#         project_root = Path(__file__).resolve().parents[1]
        
#         if config_path is None:
#             config_path = str(project_root / "data" / "config.txt")
#         if queries_path is None:
#             queries_path = str(project_root / "data" / "queries.txt")
        
#         try:
#             self.baseline_retriever = BaselineRetriever(
#                 config_path=config_path,
#                 queries_path=queries_path
#             )
#             print("Baseline Retriever initialized")
#         except Exception as e:
#             print(f"Failed to initialize baseline retriever: {e}")
#             raise
        
#         if EMBEDDINGS_AVAILABLE:
#             try:
#                 from Create_kg import load_config
#                 uri, user, password = load_config(config_path)
                
#                 self.embedding_retriever_minilm = EmbeddingRetriever(uri, user, password, "minilm")
#                 print("MiniLM Embedding Retriever initialized")
                
#                 self.embedding_retriever_mpnet = EmbeddingRetriever(uri, user, password, "mpnet")
#                 print("MPNet Embedding Retriever initialized")
#             except Exception as e:
#                 print(f"Failed to initialize embedding retrievers: {e}")
#                 self.embedding_retriever_minilm = None
#                 self.embedding_retriever_mpnet = None
#         else:
#             self.embedding_retriever_minilm = None
#             self.embedding_retriever_mpnet = None
        
#         try:
#             self.extractor = EnhancedEntityExtractor()
#             print("EnhancedEntityExtractor initialized")
#         except Exception as e:
#             print("Failed to initialize EnhancedEntityExtractor:", e)
#             self.extractor = None
    
#     def close(self):
#         if hasattr(self, 'baseline_retriever'):
#             self.baseline_retriever.close()
#         if hasattr(self, 'embedding_retriever_minilm') and self.embedding_retriever_minilm:
#             self.embedding_retriever_minilm.close()
#         if hasattr(self, 'embedding_retriever_mpnet') and self.embedding_retriever_mpnet:
#             self.embedding_retriever_mpnet.close()
    
#     def extract_intent(self, query: str) -> Dict[str, Any]:
#         print("\nStep 1: Classifying Intent...")
#         result = hybrid_intent_detection(query)
#         intent = result.get("intent", "unknown")
#         method = result.get("method", "unknown")
#         reason = result.get("reason", "No reason provided")
#         print(f"   Initial intent: {intent} (method={method}) - {reason}")
#         return result
    
#     def extract_entities_from_query(self, query: str) -> Dict[str, Any]:
#         print("\nStep 2: Extracting Entities...")

#         if self.extractor:
#             entities = self.extractor.extract_all(query)
#         else:
#             entities = {
#                 "hotels": [], "cities": [], "countries": [], "traveler_types": [],
#                 "facilities": [], "facility_ratings": {}, "nationality": [],
#                 "age_numbers": [], "gender": [], "star_rating": None, "min_rating": None,
#                 "min_facility_score": None, "min_reviews": None, "limit": 5,
#                 "min_score": None, "max_score": None, "min_location_score": None
#             }

#         entities["limit"] = 5

#         formatted = format_entities_output(entities)
#         if not formatted.startswith("No entities detected"):
#             print(f"\n{formatted}")
#         else:
#             print("   No entities detected")
        
#         return entities
    
#     def map_entities_to_params(self, intent: str, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
#         print("\nStep 3: Mapping Entities to Query Parameters...")
        
#         if self.extractor:
#             params = self.extractor.map_entities_to_params(intent, entities, query)
#         else:
#             params = {}
        
#         params["limit"] = 5
        
#         if intent == "hotels_by_score_range":
#             params["min_score"] = entities.get("min_score", 0.0)
#             params["max_score"] = entities.get("max_score", 10.0)
        
#         if intent == "hotels_by_location_score":
#             params["min_location_score"] = entities.get("min_location_score", 9.0)
        
#         for key, value in params.items():
#             print(f"   {key}: {value}")
        
#         if not params or (len(params) == 1 and "limit" in params):
#             print("   No specific parameters extracted, using defaults")
        
#         return params
    
#     def normalize_intent(self, intent: str) -> str:
#         intent_mapping = {
#             "hotel_search": "recommendation",
#             "hotel_details": "hotel_search",
#             "hotel_reviews": "review_lookup",
#             "hotel_recommendation": "recommendation",
#             "amenity_filtering": "facility_search",
#             "location_query": "hotel_search",
#             "general_question": "hotel_search",
#             "traveller_query": "traveller_type_preferences",
#             "traveller_type_preferences": "traveller_type_preferences",
#             "hotels_with_min_reviews": "hotels_with_min_reviews",
#             "hotels_by_traveler_gender_age": "hotels_by_traveler_gender_age",
#             "score_filtering": "hotels_by_score_range",
#             "hotels_by_score_range": "hotels_by_score_range",
#             "best_value_hotels": "best_value_hotels",
#             "hotels_by_location_score": "hotels_by_location_score",
#             "hotels_with_best_staff": "hotels_with_best_staff",
#             "hotels_by_traveler_age_range": "hotels_by_traveler_age_range",
#             "hotels_by_cleanliness_and_reviews": "hotels_by_cleanliness_and_reviews"  # NEW
#         }
#         return intent_mapping.get(intent, "hotel_search")    
    
#     def retrieve_baseline_results(self, intent: str, params: Dict[str, Any]) -> List[Dict]:
#         print("\nStep 4A: Querying Database (Baseline)...")
        
#         retriever_intent = self.normalize_intent(intent)
#         print(f"   Using retriever intent: {retriever_intent}")
        
#         try:
#             results = self.baseline_retriever.retrieve(retriever_intent, params)
#             print(f"   Retrieved {len(results)} baseline results")
#             return results[:3] if results else []
#         except Exception as e:
#             print(f"   Query failed: {e}")
#             return []
    
#     def retrieve_embedding_results(self, query: str, entities: Dict[str, Any]) -> Dict[str, List[Dict]]:
#         print("\nStep 4B: Querying with Embeddings...")
        
#         if not self.embedding_retriever_minilm and not self.embedding_retriever_mpnet:
#             print("   Embedding retrievers not available")
#             return {}
        
#         city_filter = None
#         if entities.get("cities"):
#             city_filter = entities["cities"][0]
#             if city_filter:
#                 city_filter = city_filter.title()
        
#         results = {}
        
#         if self.embedding_retriever_minilm:
#             print(f"   Searching with MiniLM...")
#             try:
#                 minilm_results = self.embedding_retriever_minilm.search(
#                     query, 
#                     limit=5,
#                     city_filter=city_filter,
#                     auto_detect_city=True
#                 )
#                 results["minilm"] = minilm_results[:3] if minilm_results else []
#                 print(f"   Retrieved {len(results['minilm'])} MiniLM results")
#             except Exception as e:
#                 print(f"   MiniLM search failed: {e}")
#                 results["minilm"] = []
        
#         if self.embedding_retriever_mpnet:
#             print(f"   Searching with MPNet...")
#             try:
#                 mpnet_results = self.embedding_retriever_mpnet.search(
#                     query,
#                     limit=5,
#                     city_filter=city_filter,
#                     auto_detect_city=True
#                 )
#                 results["mpnet"] = mpnet_results[:3] if mpnet_results else []
#                 print(f"   Retrieved {len(results['mpnet'])} MPNet results")
#             except Exception as e:
#                 print(f"   MPNet search failed: {e}")
#                 results["mpnet"] = []
        
#         return results
    
#     #  def format_results_structured(self, baseline_results: List[Dict], 
#     #                               embedding_results: Dict[str, List[Dict]], 
#     #                               intent: str) -> Dict[str, Any]:
        
#     #     output = {
#     #         "baseline": {"nodes": [], "reviews": []},
#     #         "minilm": {"nodes": [], "reviews": []},
#     #         "mpnet": {"nodes": [], "reviews": []}
#     #     }
        
#     #     if intent in ["hotel_reviews", "review_lookup"]:
#     #         output["baseline"]["reviews"] = baseline_results[:5] if baseline_results else []
#     #         return output
        
#     #     if baseline_results:
#     #         result = baseline_results[0]
#     #         node = {
#     #             "hotel_id": result.get("hotel_id"),
#     #             "name": result.get("name"),
#     #             "city": result.get("city"),
#     #             "country": result.get("country"),
#     #             "star_rating": result.get("star_rating"),
#     #             "cleanliness": result.get("cleanliness") or result.get("cleanliness_base")
#     #         }
            
#     #         reviews = result.get("reviews", [])[:5]
            
#     #         output["baseline"]["nodes"] = [node]
#     #         output["baseline"]["reviews"] = reviews
        
#     #     for model_name in ["minilm", "mpnet"]:
#     #         if model_name in embedding_results and embedding_results[model_name]:
#     #             result = embedding_results[model_name][0]
#     #             node = {
#     #                 "hotel_id": result.get("hotel_id"),
#     #                 "name": result.get("name"),
#     #                 "city": result.get("city"),
#     #                 "country": result.get("country"),
#     #                 "star_rating": result.get("star_rating"),
#     #                 "cleanliness": result.get("cleanliness_base"),
#     #                 "similarity_score": result.get("score")
#     #             }
                
#     #             hotel_id = result.get("hotel_id")
#     #             if hotel_id:
#     #                 reviews_map = self.baseline_retriever.get_reviews_for_hotels([hotel_id], limit=5)
#     #                 reviews = reviews_map.get(hotel_id, [])
#     #             else:
#     #                 reviews = []
                
#     #             output[model_name]["nodes"] = [node]
#     #             output[model_name]["reviews"] = reviews
        
#     #     return output
    

#     def format_results_structured(self, baseline_results: List[Dict], 
#                                   embedding_results: Dict[str, List[Dict]], 
#                                   intent: str) -> Dict[str, Any]:
#         print("\n[5/5] Formatting Results...")
        
#         output = {
#             "baseline": {"nodes": [], "reviews": []},
#             "minilm": {"nodes": [], "reviews": []},
#             "mpnet": {"nodes": [], "reviews": []}
#         }
        
#         # Special handling for review lookups
#         if intent in ["hotel_reviews", "review_lookup"]:
#             output["baseline"]["reviews"] = baseline_results[:10] if baseline_results else []
#             print(f"      Baseline: {len(output['baseline']['reviews'])} reviews (review lookup)")
#             return output
        
#         # Process ALL baseline results (not just the first one!)
#         if baseline_results:
#             print(f"      Processing {len(baseline_results)} baseline results...")
#             for idx, result in enumerate(baseline_results):
#                 node = {
#                     "hotel_id": result.get("hotel_id"),
#                     "name": result.get("name"),
#                     "city": result.get("city"),
#                     "country": result.get("country"),
#                     "star_rating": result.get("star_rating"),
#                     "cleanliness": result.get("cleanliness") or result.get("cleanliness_base"),
#                     "overall_rating": result.get("overall_rating"),
#                     "location_score": result.get("location_score"),
#                     "comfort": result.get("comfort") or result.get("comfort_base"),
#                     "facilities": result.get("facilities") or result.get("facilities_base"),
#                     "staff": result.get("staff") or result.get("staff_base"),
#                     "value_for_money": result.get("value_for_money") or result.get("value_for_money_base")
#                 }
                
#                 # Get reviews for this hotel - check if they're nested in the result
#                 reviews = result.get("reviews", [])
#                 if not reviews:
#                     # If no reviews in result, fetch them separately
#                     hotel_id = result.get("hotel_id")
#                     if hotel_id:
#                         reviews_map = self.baseline_retriever.get_reviews_for_hotels([hotel_id], limit=3)
#                         reviews = reviews_map.get(hotel_id, [])
                
#                 output["baseline"]["nodes"].append(node)
#                 # Limit reviews per hotel to 3, but process all hotels
#                 output["baseline"]["reviews"].extend(reviews[:3])
                
#                 print(f"        • Hotel {idx+1}: {node['name']} ({len(reviews[:3])} reviews)")
            
#             print(f"      ✓ Baseline: {len(output['baseline']['nodes'])} hotels, {len(output['baseline']['reviews'])} reviews")
#         else:
#             print(f"      ⚠ No baseline results to format")
        
#         # Process ALL embedding results (not just the first one!)
#         for model_name in ["minilm", "mpnet"]:
#             if model_name in embedding_results and embedding_results[model_name]:
#                 print(f"      Processing {len(embedding_results[model_name])} {model_name} results...")
                
#                 for idx, result in enumerate(embedding_results[model_name]):
#                     node = {
#                         "hotel_id": result.get("hotel_id"),
#                         "name": result.get("name"),
#                         "city": result.get("city"),
#                         "country": result.get("country"),
#                         "star_rating": result.get("star_rating"),
#                         "cleanliness": result.get("cleanliness_base"),
#                         "similarity_score": result.get("score")
#                     }
                    
#                     # Get reviews for this hotel
#                     hotel_id = result.get("hotel_id")
#                     if hotel_id:
#                         reviews_map = self.baseline_retriever.get_reviews_for_hotels([hotel_id], limit=3)
#                         reviews = reviews_map.get(hotel_id, [])
#                     else:
#                         reviews = []
                    
#                     output[model_name]["nodes"].append(node)
#                     output[model_name]["reviews"].extend(reviews)
                    
#                     print(f"        • Hotel {idx+1}: {node['name']} (sim: {node.get('similarity_score', 0):.3f}, {len(reviews)} reviews)")
                
#                 print(f"      ✓ {model_name.upper()}: {len(output[model_name]['nodes'])} hotels, {len(output[model_name]['reviews'])} reviews")
        
#         return output

#     def process_query(self, query: str) -> Dict[str, Any]:
#         print(f"\n{'='*80}")
#         print(f"PROCESSING QUERY: {query}")
#         print(f"{'='*80}")
        
#         intent_result = self.extract_intent(query)
#         raw_intent = intent_result.get("intent", "unknown")
#         entities = self.extract_entities_from_query(query)
#         intent = refine_intent_with_entities(raw_intent, entities, query)
#         print(f"\n   Final intent (after refinement): {intent}")
#         params = self.map_entities_to_params(intent, entities, query)

#         if intent == "hotel_reviews" and not params.get("hotel_id") and params.get("hotel_name"):
#             qname = params["hotel_name"]
#             try:
#                 tpl = self.baseline_retriever.queries.get("hotel_by_name")
#                 if tpl:
#                     hits = self.baseline_retriever.run_query(tpl, {"q": qname, "limit": 5})
#                     if hits:
#                         params["hotel_id"] = hits[0].get("hotel_id")
#                         print(f"   Resolved hotel_name '{qname}' -> hotel_id '{params['hotel_id']}'")
#             except Exception as e:
#                 print(f"   Warning: failed to resolve hotel_name to id: {e}")

#         baseline_results = self.retrieve_baseline_results(intent, params)
        
#         embedding_results = {}
#         if intent not in ["hotel_reviews", "review_lookup"]:
#             embedding_results = self.retrieve_embedding_results(query, entities)
        
#         structured_results = self.format_results_structured(baseline_results, embedding_results, intent)
        
#         output = {
#             "query": query,
#             "results": structured_results
#         }
        
#         print(f"\n{'='*80}")
#         print(f"RESULTS")
#         print(f"{'='*80}")
#         print(f"Baseline: {len(structured_results['baseline']['nodes'])} hotels, {len(structured_results['baseline']['reviews'])} reviews")
#         print(f"MiniLM: {len(structured_results['minilm']['nodes'])} hotels, {len(structured_results['minilm']['reviews'])} reviews")
#         print(f"MPNet: {len(structured_results['mpnet']['nodes'])} hotels, {len(structured_results['mpnet']['reviews'])} reviews")
#         print(f"\n{json.dumps(output, indent=2)}")
        
#         return output
    
# def retrieve_hotels(query: str, config_path: str = None, queries_path: str = None) -> Dict[str, Any]:
#     """
#     Main entry point for hotel retrieval.
#     Creates a pipeline, processes the query, and returns structured results.
#     """
#     print("\n" + "="*80)
#     print("HOTEL RETRIEVAL SYSTEM")
#     print("="*80)
    
#     pipeline = HybridHotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
#     try:
#         result = pipeline.process_query(query)
        
#         # Verify results before returning
#         print("\n" + "="*80)
#         print("VERIFICATION")
#         print("="*80)
#         baseline_count = len(result.get("results", {}).get("baseline", {}).get("nodes", []))
#         minilm_count = len(result.get("results", {}).get("minilm", {}).get("nodes", []))
#         mpnet_count = len(result.get("results", {}).get("mpnet", {}).get("nodes", []))
        
#         print(f"✓ Returning {baseline_count + minilm_count + mpnet_count} total hotels")
#         print(f"  - Baseline: {baseline_count}")
#         print(f"  - MiniLM: {minilm_count}")
#         print(f"  - MPNet: {mpnet_count}")
        
#         if baseline_count == 0 and minilm_count == 0 and mpnet_count == 0:
#             print("\n⚠ WARNING: No results found from any retriever!")
        
#         return result
#     except Exception as e:
#         print(f"\n✗ ERROR in retrieve_hotels: {e}")
#         import traceback
#         traceback.print_exc()
#         raise
#     finally:
#         pipeline.close()

# # def retrieve_hotels(query: str, config_path: str = None, queries_path: str = None) -> Dict[str, Any]:
# #     pipeline = HybridHotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
# #     try:
# #         result = pipeline.process_query(query)
# #         return result
# #     finally:
# #         pipeline.close()

# def interactive_mode(config_path=None, queries_path=None):
#     print("\n" + "="*80)
#     print("HYBRID HOTEL SEARCH PIPELINE - INTERACTIVE MODE")
#     print("="*80)
#     print("\nType your queries below. Type 'exit' or 'quit' to stop.\n")
    
#     pipeline = HybridHotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
#     try:
#         while True:
#             try:
#                 query = input("\nEnter your query: ").strip()
#                 if not query:
#                     continue
#                 if query.lower() in ['exit', 'quit', 'q']:
#                     print("\nGoodbye!")
#                     break
#                 pipeline.process_query(query)
#             except KeyboardInterrupt:
#                 print("\n\nGoodbye!")
#                 break
#             except Exception as e:
#                 print(f"\nError processing query: {e}")
#                 import traceback
#                 traceback.print_exc()
#     finally:
#         pipeline.close()


# def batch_mode(queries_file: str, config_path=None, queries_path=None):
#     print("\n" + "="*80)
#     print("HYBRID HOTEL SEARCH PIPELINE - BATCH MODE")
#     print("="*80)
    
#     queries = []
#     with open(queries_file, 'r', encoding='utf-8') as f:
#         queries = [line.strip() for line in f if line.strip()]
    
#     pipeline = HybridHotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
#     try:
#         for i, query in enumerate(queries, 1):
#             print(f"\n\n{'#'*80}")
#             print(f"# QUERY {i}/{len(queries)}")
#             print(f"{'#'*80}")
#             pipeline.process_query(query)
#     finally:
#         pipeline.close()


# def main():
#     # import argparse
    
#     # parser = argparse.ArgumentParser(
#     #     description="Hybrid Hotel Search Pipeline - Baseline + Embeddings"
#     # )
#     # parser.add_argument(
#     #     '--mode',
#     #     choices=['interactive', 'batch'],
#     #     default='interactive',
#     #     help='Execution mode (default: interactive)'
#     # )
#     # parser.add_argument(
#     #     '--config',
#     #     type=str,
#     #     default=None,
#     #     help='Path to config.txt'
#     # )
#     # parser.add_argument(
#     #     '--queries',
#     #     type=str,
#     #     default=None,
#     #     help='Path to queries.txt'
#     # )
#     # parser.add_argument(
#     #     '--batch-file',
#     #     type=str,
#     #     default=None,
#     #     help='Path to file containing queries (one per line) for batch mode'
#     # )
    
#     # args = parser.parse_args()
    
#     # if args.mode == 'interactive':
#     #     interactive_mode(config_path=args.config, queries_path=args.queries)
#     # elif args.mode == 'batch':
#     #     if not args.batch_file:
#     #         print("Error: --batch-file required for batch mode")
#     #         return
#     #     batch_mode(args.batch_file, config_path=args.config, queries_path=args.queries)
#     test_queries = "Find hotels for solo travellers aged 20-24"
    
    
#     print("Testing retrieve_hotels function...")
#     result = retrieve_hotels(test_queries)
    
#     print("\n" + "="*80)
#     print("FINAL OUTPUT:")
#     print("="*80)
#     print(json.dumps(result, indent=2))


# if __name__ == "__main__":
#     main()


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

try:
    from embeddings_retreiver import EmbeddingRetriever
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: embeddings_retreiver not available, running baseline only")


class HybridHotelSearchPipeline:
    def __init__(self, config_path: str = None, queries_path: str = None):
        print("Initializing Hybrid Hotel Search Pipeline...")
        
        project_root = Path(__file__).resolve().parents[1]
        
        if config_path is None:
            config_path = str(project_root / "data" / "config.txt")
        if queries_path is None:
            queries_path = str(project_root / "data" / "queries.txt")
        
        try:
            self.baseline_retriever = BaselineRetriever(
                config_path=config_path,
                queries_path=queries_path
            )
            print("✓ Baseline Retriever initialized")
        except Exception as e:
            print(f"✗ Failed to initialize baseline retriever: {e}")
            raise
        
        if EMBEDDINGS_AVAILABLE:
            try:
                from Create_kg import load_config
                uri, user, password = load_config(config_path)
                
                self.embedding_retriever_minilm = EmbeddingRetriever(uri, user, password, "minilm")
                print("✓ MiniLM Embedding Retriever initialized")
                
                self.embedding_retriever_mpnet = EmbeddingRetriever(uri, user, password, "mpnet")
                print("✓ MPNet Embedding Retriever initialized")
            except Exception as e:
                print(f"✗ Failed to initialize embedding retrievers: {e}")
                self.embedding_retriever_minilm = None
                self.embedding_retriever_mpnet = None
        else:
            self.embedding_retriever_minilm = None
            self.embedding_retriever_mpnet = None
        
        try:
            self.extractor = EnhancedEntityExtractor()
            print("✓ EnhancedEntityExtractor initialized")
        except Exception as e:
            print(f"✗ Failed to initialize EnhancedEntityExtractor: {e}")
            self.extractor = None
    
    def close(self):
        if hasattr(self, 'baseline_retriever'):
            self.baseline_retriever.close()
        if hasattr(self, 'embedding_retriever_minilm') and self.embedding_retriever_minilm:
            self.embedding_retriever_minilm.close()
        if hasattr(self, 'embedding_retriever_mpnet') and self.embedding_retriever_mpnet:
            self.embedding_retriever_mpnet.close()
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        print("\n[1/5] Classifying Intent...")
        result = hybrid_intent_detection(query)
        intent = result.get("intent", "unknown")
        method = result.get("method", "unknown")
        reason = result.get("reason", "No reason provided")
        print(f"      Initial intent: {intent} (method={method})")
        print(f"      Reason: {reason}")
        return result
    
    def extract_entities_from_query(self, query: str) -> Dict[str, Any]:
        print("\n[2/5] Extracting Entities...")

        if self.extractor:
            entities = self.extractor.extract_all(query)
        else:
            entities = {
                "hotels": [], "cities": [], "countries": [], "traveler_types": [],
                "facilities": [], "facility_ratings": {}, "nationality": [],
                "age_numbers": [], "gender": [], "star_rating": None, "min_rating": None,
                "min_facility_score": None, "min_reviews": None, "limit": 5,
                "min_score": None, "max_score": None, "min_location_score": None
            }

        # Increase limit for better results
        entities["limit"] = 5

        formatted = format_entities_output(entities)
        if not formatted.startswith("No entities detected"):
            print(f"{formatted}")
        else:
            print("      No entities detected")
        
        return entities
    
    def map_entities_to_params(self, intent: str, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        print("\n[3/5] Mapping Entities to Query Parameters...")
        
        if self.extractor:
            params = self.extractor.map_entities_to_params(intent, entities, query)
        else:
            params = {}
        
        # Increase limit
        params["limit"] = 5
        
        if intent == "hotels_by_score_range":
            params["min_score"] = entities.get("min_score", 0.0)
            params["max_score"] = entities.get("max_score", 10.0)
        
        if intent == "hotels_by_location_score":
            params["min_location_score"] = entities.get("min_location_score", 9.0)
        
        print("      Parameters:")
        for key, value in params.items():
            print(f"      • {key}: {value}")
        
        if not params or (len(params) == 1 and "limit" in params):
            print("      ⚠ No specific parameters extracted, using defaults")
        
        return params
    
    def normalize_intent(self, intent: str) -> str:
        intent_mapping = {
            "hotel_search": "recommendation",
            "hotel_details": "hotel_search",
            "hotel_reviews": "review_lookup",
            "hotel_recommendation": "recommendation",
            "amenity_filtering": "facility_search",
            "location_query": "hotel_search",
            "general_question": "hotel_search",
            "traveller_query": "traveller_type_preferences",
            "traveller_type_preferences": "traveller_type_preferences",
            "hotels_with_min_reviews": "hotels_with_min_reviews",
            "hotels_by_traveler_gender_age": "hotels_by_traveler_gender_age",
            "score_filtering": "hotels_by_score_range",
            "hotels_by_score_range": "hotels_by_score_range",
            "best_value_hotels": "best_value_hotels",
            "hotels_by_location_score": "hotels_by_location_score",
            "hotels_with_best_staff": "hotels_with_best_staff"
        }
        return intent_mapping.get(intent, "hotel_search")
    
    def retrieve_baseline_results(self, intent: str, params: Dict[str, Any]) -> List[Dict]:
        print("\n[4/5] Querying Database (Baseline Retriever)...")
        
        retriever_intent = self.normalize_intent(intent)
        print(f"      Using retriever intent: {retriever_intent}")
        print(f"      Parameters: {params}")
        
        try:
            results = self.baseline_retriever.retrieve(retriever_intent, params)
            print(f"      ✓ Retrieved {len(results)} baseline results")
            
            # Debug: Print first result structure
            if results:
                print(f"      First result keys: {list(results[0].keys())}")
                first_hotel_name = results[0].get('name', 'Unknown')
                first_hotel_id = results[0].get('hotel_id', 'N/A')
                print(f"      First hotel: {first_hotel_name} (ID: {first_hotel_id})")
            else:
                print(f"      ⚠ No results returned from baseline retriever")
            
            # Return ALL results, don't limit here - let format_results handle it
            return results if results else []
        except Exception as e:
            print(f"      ✗ Query failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def retrieve_embedding_results(self, query: str, entities: Dict[str, Any]) -> Dict[str, List[Dict]]:
        print("\n[4/5] Querying with Embeddings...")
        
        if not self.embedding_retriever_minilm and not self.embedding_retriever_mpnet:
            print("      ⚠ Embedding retrievers not available")
            return {}
        
        city_filter = None
        if entities.get("cities"):
            city_filter = entities["cities"][0]
            if city_filter:
                city_filter = city_filter.title()
        
        results = {}
        
        if self.embedding_retriever_minilm:
            print(f"      Searching with MiniLM (city_filter={city_filter})...")
            try:
                minilm_results = self.embedding_retriever_minilm.search(
                    query, 
                    limit=5,
                    city_filter=city_filter,
                    auto_detect_city=True
                )
                # Return ALL results, don't limit here
                results["minilm"] = minilm_results if minilm_results else []
                print(f"      ✓ Retrieved {len(results['minilm'])} MiniLM results")
                if results["minilm"]:
                    print(f"        First MiniLM result: {results['minilm'][0].get('name', 'Unknown')}")
            except Exception as e:
                print(f"      ✗ MiniLM search failed: {e}")
                results["minilm"] = []
        
        if self.embedding_retriever_mpnet:
            print(f"      Searching with MPNet (city_filter={city_filter})...")
            try:
                mpnet_results = self.embedding_retriever_mpnet.search(
                    query,
                    limit=5,
                    city_filter=city_filter,
                    auto_detect_city=True
                )
                # Return ALL results, don't limit here
                results["mpnet"] = mpnet_results if mpnet_results else []
                print(f"      ✓ Retrieved {len(results['mpnet'])} MPNet results")
                if results["mpnet"]:
                    print(f"        First MPNet result: {results['mpnet'][0].get('name', 'Unknown')}")
            except Exception as e:
                print(f"      ✗ MPNet search failed: {e}")
                results["mpnet"] = []
        
        return results
    
    def format_results_structured(self, baseline_results: List[Dict], 
                              embedding_results: Dict[str, List[Dict]], 
                              intent: str) -> Dict[str, Any]:
        print("\n[5/5] Formatting Results...")
        
        output = {
            "baseline": {"nodes": [], "reviews": []},
            "minilm": {"nodes": [], "reviews": []},
            "mpnet": {"nodes": [], "reviews": []}
        }
        
        # Special handling for review lookups
        if intent in ["hotel_reviews", "review_lookup"]:
            output["baseline"]["reviews"] = baseline_results[:10] if baseline_results else []
            print(f"      Baseline: {len(output['baseline']['reviews'])} reviews (review lookup)")
            return output
        
        # Process ALL baseline results (not just the first one!)
        if baseline_results:
            print(f"      Processing {len(baseline_results)} baseline results...")
            for idx, result in enumerate(baseline_results):
                # Get reviews for this hotel - check if they're nested in the result
                reviews = result.get("reviews", [])
                if not reviews:
                    # If no reviews in result, fetch them separately
                    hotel_id = result.get("hotel_id")
                    if hotel_id:
                        reviews_map = self.baseline_retriever.get_reviews_for_hotels([hotel_id], limit=3)
                        reviews = reviews_map.get(hotel_id, [])
                
                # IMPORTANT: Only include non-null fields
                node = {
                    "hotel_id": result.get("hotel_id"),
                    "name": result.get("name"),
                    "city": result.get("city"),
                    "country": result.get("country"),
                    "star_rating": result.get("star_rating"),
                }
                
                # Add optional fields only if they exist and are not None
                optional_fields = {
                    "cleanliness": result.get("cleanliness") or result.get("cleanliness_base"),
                    "overall_rating": result.get("overall_rating"),
                    "location_score": result.get("location_score"),
                    "comfort": result.get("comfort") or result.get("comfort_base"),
                    "facilities": result.get("facilities") or result.get("facilities_base"),
                    "staff": result.get("staff") or result.get("staff_base"),
                    "value_for_money": result.get("value_for_money") or result.get("value_for_money_base")
                }
                
                # Only add fields that are not None
                for key, value in optional_fields.items():
                    if value is not None:
                        node[key] = value
                
                # Add reviews (keep reviews with the hotel!)
                node["reviews"] = reviews[:3]
                
                output["baseline"]["nodes"].append(node)
                
                print(f"        • Hotel {idx+1}: {node['name']} ({len(reviews[:3])} reviews)")
            
            print(f"      ✓ Baseline: {len(output['baseline']['nodes'])} hotels")
            # Remove the separate reviews list since reviews are now in nodes
            output["baseline"]["reviews"] = []
        else:
            print(f"      ⚠ No baseline results to format")
        
        # Process ALL embedding results (not just the first one!)
        for model_name in ["minilm", "mpnet"]:
            if model_name in embedding_results and embedding_results[model_name]:
                print(f"      Processing {len(embedding_results[model_name])} {model_name} results...")
                
                for idx, result in enumerate(embedding_results[model_name]):
                    # Get reviews for this hotel
                    hotel_id = result.get("hotel_id")
                    if hotel_id:
                        reviews_map = self.baseline_retriever.get_reviews_for_hotels([hotel_id], limit=3)
                        reviews = reviews_map.get(hotel_id, [])
                    else:
                        reviews = []
                    
                    # IMPORTANT: Include reviews directly in the node object
                    node = {
                        "hotel_id": result.get("hotel_id"),
                        "name": result.get("name"),
                        "city": result.get("city"),
                        "country": result.get("country"),
                        "star_rating": result.get("star_rating"),
                        "similarity_score": result.get("score"),
                        "reviews": reviews[:3]
                    }
                    
                    # Add cleanliness only if not None
                    if result.get("cleanliness_base") is not None:
                        node["cleanliness"] = result.get("cleanliness_base")
                    
                    output[model_name]["nodes"].append(node)
                    
                    print(f"        • Hotel {idx+1}: {node['name']} (sim: {node.get('similarity_score', 0):.3f}, {len(reviews[:3])} reviews)")
                
                print(f"      ✓ {model_name.upper()}: {len(output[model_name]['nodes'])} hotels")
                # Remove the separate reviews list since reviews are now in nodes
                output[model_name]["reviews"] = []
        
        return output
    
    def process_query(self, query: str) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"PROCESSING QUERY: {query}")
        print(f"{'='*80}")
        
        # Step 1: Extract intent
        intent_result = self.extract_intent(query)
        raw_intent = intent_result.get("intent", "unknown")
        
        # Step 2: Extract entities
        entities = self.extract_entities_from_query(query)
        
        # Refine intent with entities
        intent = refine_intent_with_entities(raw_intent, entities, query)
        print(f"\n      ✓ Final intent (after refinement): {intent}")
        
        # Step 3: Map entities to parameters
        params = self.map_entities_to_params(intent, entities, query)

        # Handle hotel name to ID resolution for review lookups
        if intent == "hotel_reviews" and not params.get("hotel_id") and params.get("hotel_name"):
            qname = params["hotel_name"]
            try:
                tpl = self.baseline_retriever.queries.get("hotel_by_name")
                if tpl:
                    hits = self.baseline_retriever.run_query(tpl, {"q": qname, "limit": 1})
                    if hits:
                        params["hotel_id"] = hits[0].get("hotel_id")
                        print(f"      ✓ Resolved hotel_name '{qname}' -> hotel_id '{params['hotel_id']}'")
            except Exception as e:
                print(f"      ⚠ Warning: failed to resolve hotel_name to id: {e}")

        # Step 4: Retrieve results
        baseline_results = self.retrieve_baseline_results(intent, params)
        
        embedding_results = {}
        if intent not in ["hotel_reviews", "review_lookup"]:
            embedding_results = self.retrieve_embedding_results(query, entities)
        
        # Step 5: Format results
        structured_results = self.format_results_structured(baseline_results, embedding_results, intent)
        
        output = {
            "query": query,
            "results": structured_results
        }
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Baseline:  {len(structured_results['baseline']['nodes'])} hotels, {len(structured_results['baseline']['reviews'])} reviews")
        print(f"MiniLM:    {len(structured_results['minilm']['nodes'])} hotels, {len(structured_results['minilm']['reviews'])} reviews")
        print(f"MPNet:     {len(structured_results['mpnet']['nodes'])} hotels, {len(structured_results['mpnet']['reviews'])} reviews")
        
        return output

def retrieve_hotels(query: str, config_path: str = None, queries_path: str = None) -> Dict[str, Any]:
    """
    Main entry point for hotel retrieval.
    Creates a pipeline, processes the query, and returns structured results.
    """
    print("\n" + "="*80)
    print("HOTEL RETRIEVAL SYSTEM")
    print("="*80)
    
    pipeline = HybridHotelSearchPipeline(config_path=config_path, queries_path=queries_path)
    
    try:
        result = pipeline.process_query(query)
        
        # Verify results before returning
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        baseline_count = len(result.get("results", {}).get("baseline", {}).get("nodes", []))
        minilm_count = len(result.get("results", {}).get("minilm", {}).get("nodes", []))
        mpnet_count = len(result.get("results", {}).get("mpnet", {}).get("nodes", []))
        
        print(f"✓ Returning {baseline_count + minilm_count + mpnet_count} total hotels")
        print(f"  - Baseline: {baseline_count}")
        print(f"  - MiniLM: {minilm_count}")
        print(f"  - MPNet: {mpnet_count}")
        
        if baseline_count == 0 and minilm_count == 0 and mpnet_count == 0:
            print("\n⚠ WARNING: No results found from any retriever!")
        
        return result
    except Exception as e:
        print(f"\n✗ ERROR in retrieve_hotels: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        pipeline.close()


def main():
    test_query = "Show me The Royal Compass"
    
    print("Testing retrieve_hotels function...")
    result = retrieve_hotels(test_query)
    
    print("\n" + "="*80)
    print("FINAL JSON OUTPUT:")
    print("="*80)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()