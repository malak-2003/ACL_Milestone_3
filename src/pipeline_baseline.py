from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re
import json
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

# Try to import user modules. Adjust import paths if your files have different names.
try:
    # intent classifier exports: hybrid_intent_detection (and classify_intent_llm)
    from Preprocessing.intent_classifier import hybrid_intent_detection
except Exception as e:
    logger.exception("Failed to import hybrid_intent_detection from preprocessing.intent_classifier: %s", e)
    raise

try:
    # entities_extraction exports: extract_entities, format_entities_output
    from Preprocessing.entities_extraction import extract_entities, format_entities_output
except Exception as e:
    logger.exception("Failed to import extract_entities from preprocessing.entities_extraction: %s", e)
    raise

try:
    from baseline_retreiver import BaselineRetriever
except Exception as e:
    logger.exception("Failed to import BaselineRetriever from baseline_retreiver: %s", e)
    raise

# -----------------------------
# Helpers: map intents + build retriever params
# -----------------------------
# Map the intents your classifier returns -> retriever expected intents
INTENT_MAP = {
    # your classifier intents (examples from your prompt)
    "hotel_search": "hotel_search",
    "hotel_details": "hotel_search",        # use hotel_search/hotel_by_name template
    "hotel_reviews": "review_lookup",       # maps to review_lookup
    "hotel_recommendation": "recommendation",
    "amenity_filtering": "facility_search",
    "location_query": "hotel_search",
    "visa_requirements": "visa_query",
    "general_question": "hotel_search",
    "unknown": "hotel_search"
}

NUMERIC_EXTRACT_RE = re.compile(r"(\d+(?:\.\d+)?)")

def first_or_none(lst: List[Any]) -> Optional[Any]:
    return lst[0] if lst else None

def extract_price_and_rating_from_text(text: str) -> Dict[str, Any]:
    """
    Heuristic extraction for price and rating from free text.
    Returns keys: max_price (int), min_price (int), star (int), min_rating (float)
    """
    out = {}
    t = text.lower()
    # price patterns: "$200", "under 200", "below 200", "less than 200", "under $200"
    m = re.search(r"(?:under|below|less than|under \$|<)\s*\$?(\d+(?:\.\d+)?)", t)
    if m:
        out["max_price"] = int(float(m.group(1)))
    else:
        # "under $200" could also be "below 200"
        m2 = re.search(r"\$?(\d+(?:\.\d+)?)\s*(?:dollars|usd)?\b", t)
        if m2 and ("hotel" in t or "find" in t or "price" in t or "under" not in t and "between" in t):
            # be conservative: only set when explicit price words present or "under" matched above
            pass

    # star rating patterns: "4 star", "4-star", "4 stars"
    m = re.search(r"(\d)\s*[-]?\s*star", t)
    if m:
        try:
            out["star"] = int(m.group(1))
        except Exception:
            pass

    # rating patterns: "rating >= 4", ">= 4", "at least 4", "minimum rating 4"
    m = re.search(r"(?:rating|rated|min(?:imum)? rating|>=|at least|minimum)\s*(?:is\s*)?(\d+(?:\.\d+)?)", t)
    if m:
        try:
            out["min_rating"] = float(m.group(1))
        except Exception:
            pass

    return out

def build_retriever_entities(extracted: Dict[str, Any], original_query: str, mapped_intent: str) -> Dict[str, Any]:
    """
    Convert the output of extract_entities(...) to the parameter dict expected by BaselineRetriever.retrieve()
    Common output keys for BaselineRetriever: city, hotel_name, hotel_id, min_rating, limit, country, min_facility_score, facility, country_from, country_to, min_reviews
    """
    params: Dict[str, Any] = {}

    # prefer explicit city entities
    city = first_or_none(extracted.get("cities", []))
    country = first_or_none(extracted.get("countries", []))

    # hotels list (from hotel dataset match)
    hotel_name = first_or_none(extracted.get("hotels", []))
    if hotel_name:
        params["hotel_name"] = hotel_name

    # If spaCy detected GPE that looks like a city, use it
    if city:
        params["city"] = city.title() 
    # If only a country was found and not city, pass country
    if not city and country:
        params["country"] = country

    # traveler type (unused by retriever but include)
    if extracted.get("traveler_types"):
        params["traveller_type"] = first_or_none(extracted["traveler_types"])

    # Ages/gender not used directly by retriever but included
    if extracted.get("age_numbers"):
        params["age_numbers"] = extracted["age_numbers"]
    if extracted.get("gender"):
        params["gender"] = extracted["gender"]

    # Look for facility keywords in the original query
    facilities = []
    facility_keywords = ["pool", "wifi", "parking", "gym", "spa", "breakfast", "restaurant", "bar", "wifi"]
    for k in facility_keywords:
        if k in original_query.lower():
            facilities.append(k)
    if facilities:
        # pass the first matching facility as retriever expects single facility param
        params["facility"] = facilities[0]

    # Try to extract min_rating / star / price heuristics
    heur = extract_price_and_rating_from_text(original_query)
    if "min_rating" in heur:
        params["min_rating"] = heur["min_rating"]
    if "star" in heur:
        # interpret star as min_rating if rating not present
        if "min_rating" not in params:
            params["min_rating"] = float(heur["star"])

    # Limit param (default)
    params["limit"] = extracted.get("limit", 15)

    # Visa-specific mapping
    if mapped_intent == "visa_query":
        # find country_from (nationality or "from <country>") and country_to (destination)
        # Simple heuristics:
        nf = first_or_none(extracted.get("nationality", []))
        if nf:
            params["country_from"] = nf
        if country:
            params["country_to"] = country
        # Try to find "to X" pattern
        m = re.search(r"(?:to|travel to|visit)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", original_query)
        if m:
            params["country_to"] = m.group(1).strip().lower()

    # Review lookup mapping
    if mapped_intent == "review_lookup":
        # If we have hotel id or name use it
        if "hotel_name" not in params and extracted.get("hotels"):
            params["hotel_name"] = extracted["hotels"][0]
        # No hotel id -> return empty and caller will handle fallback
    # Facility search mapping
    if mapped_intent == "facility_search":
        # If user asked for "min facility score" (heuristic) check for words
        m = re.search(r"(?:facility|facilities).*(?:score|rating).*?(\d+(?:\.\d+)?)", original_query.lower())
        if m:
            try:
                params["min_facility_score"] = float(m.group(1))
            except Exception:
                pass

    # If the user included 'reviews' or 'rating' but not intent mapped to review_lookup, we might promote to review_lookup
    if "review" in original_query.lower() and mapped_intent != "review_lookup":
        params.setdefault("note", "user_asked_for_reviews")

    return params

# -----------------------------
# Pipeline runner
# -----------------------------
class KGQueryPipeline:
    def __init__(self, config_path: Optional[str] = None, queries_path: Optional[str] = None):
        """
        Initialize the retriever. Paths default to project-root/data/config.txt and data/queries.txt
        """
        # If BaselineRetriever uses project_root parents[1] already, it will locate default files.
        # We still forward explicit paths if provided.
        self.retriever = BaselineRetriever(config_path=config_path, queries_path=queries_path)

    def close(self):
        self.retriever.close()

    def run(self, user_query: str, limit: int = 15) -> Dict[str, Any]:
        """
        Run the full pipeline for a single user query and return a dict containing:
          - intent (original)
          - mapped_intent (for retriever)
          - entities (raw from NER)
          - retriever_params (mapped)
          - results (list)  (may be empty)
          - diagnostics (list)
        """
        diagnostics: List[str] = []
        # 1) Intent
        intent_result = hybrid_intent_detection(user_query)
        intent = intent_result.get("intent", "unknown")
        diagnostics.append(f"intent_detection: {json.dumps(intent_result)}")

        # Map to retriever intent
        mapped_intent = INTENT_MAP.get(intent, "hotel_search")
        diagnostics.append(f"mapped_intent: {mapped_intent}")

        # 2) Entities extraction
        entities = extract_entities(user_query)
        diagnostics.append(f"entities_extracted: {entities}")

        # 3) Build retriever params
        # add limit to extracted to respect max results
        entities["_original_query"] = user_query
        entities["limit"] = limit
        retriever_params = build_retriever_entities(entities, user_query, mapped_intent)
        diagnostics.append(f"retriever_params_built: {retriever_params}")

        # 4) Call retriever
        try:
            results = self.retriever.retrieve(mapped_intent, retriever_params)
            diagnostics.append(f"retriever_call_success: returned {len(results)} rows")
        except Exception as e:
            logger.exception("Retriever call failed: %s", e)
            diagnostics.append(f"retriever_error: {str(e)}")
            results = []

        return {
            "intent": intent,
            "intent_info": intent_result,
            "mapped_intent": mapped_intent,
            "entities": entities,
            "retriever_params": retriever_params,
            "results": results,
            "diagnostics": diagnostics
        }

# -----------------------------
# CLI Demo / Example usage
# -----------------------------
DEFAULT_TEST_QUERIES = [
    "Find me hotels in Dubai under $200",
    "Does the Marriott Downtown have a swimming pool?",
    "Show me reviews for Hilton Cairo",
    "What is the distance between my hotel and the pyramids?",
    "Do Egyptians need a visa to travel to Thailand?",
    "I need a hotel with free wifi and breakfast",
    "Which hotels are closest to the airport?",
    "Tell me about the best luxury hotels in Paris"
]

def demo(config_path: Optional[str] = None, queries_path: Optional[str] = None):
    logger.info("Starting pipeline demo...")
    pipeline = KGQueryPipeline(config_path=config_path, queries_path=queries_path)

    try:
        for q in DEFAULT_TEST_QUERIES:
            print("\n" + "=" * 80)
            print("QUERY:", q)
            out = pipeline.run(q, limit=5)
            print("Intent:", out["intent"])
            print("Mapped intent:", out["mapped_intent"])
            print("Retriever params:", out["retriever_params"])
            print("Results count:", len(out["results"]))
            if out["results"]:
                # pretty print up to 3 results
                for i, r in enumerate(out["results"][:3], start=1):
                    print(f"  - sample {i}: {r}")
            else:
                print("  <-- no results returned (check Neo4j DB / queries.txt / city/hotel names).")
            # print diagnostics for debugging
            print("Diagnostics (short):", out["diagnostics"][:2])
    finally:
        pipeline.close()
        logger.info("Pipeline demo finished.")

# -----------------------------
# Exported convenience function
# -----------------------------
def run_query(user_query: str, config_path: Optional[str] = None, queries_path: Optional[str] = None, limit: int = 15) -> Dict[str, Any]:
    """
    Convenience entry point for other modules.
    Example:
        from src.pipeline import run_query
        res = run_query("Find hotels in Cairo", config_path="data/config.txt")
    """
    p = KGQueryPipeline(config_path=config_path, queries_path=queries_path)
    try:
        return p.run(user_query, limit=limit)
    finally:
        p.close()

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the KG query pipeline demo or a single query.")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo queries")
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    parser.add_argument("--config", type=str, default=None, help="Path to data/config.txt")
    parser.add_argument("--queries", type=str, default=None, help="Path to data/queries.txt")
    parser.add_argument("--limit", type=int, default=5, help="Result limit for retriever")
    parser.add_argument("--interactive", action="store_true", help="Start interactive prompt")
    args = parser.parse_args()

    if args.demo:
        demo(config_path=args.config, queries_path=args.queries)
    elif args.query:
        res = run_query(args.query, config_path=args.config, queries_path=args.queries, limit=args.limit)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        # Interactive prompt
        print("Interactive mode. Type a query and press Enter. Type 'quit' or 'exit' to stop.")
        pipeline = KGQueryPipeline(config_path=args.config, queries_path=args.queries)
        try:
            while True:
                try:
                    q = input("\nEnter query > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break
                if not q:
                    continue
                if q.lower() in ("quit", "exit"):
                    print("Goodbye.")
                    break
                out = pipeline.run(q, limit=args.limit)
                # Print a short summary
                print("\nIntent:", out["intent"])
                print("Mapped intent:", out["mapped_intent"])
                print("Retriever params:", out["retriever_params"])
                print("Results count:", len(out["results"]))
                if out["results"]:
                    for i, r in enumerate(out["results"][:5], start=1):
                        print(f"  - sample {i}: {r}")
                else:
                    print("  <-- no results returned. (Check DB / config / queries)")
        finally:
            pipeline.close()
