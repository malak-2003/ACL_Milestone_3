# src/baseline_retreiver.py
import os
import sys
from neo4j import GraphDatabase
from typing import Dict, Any
from pathlib import Path
from Create_kg import load_config

# -------------------------
# Query file loader
# -------------------------
def load_queries(path: str) -> Dict[str, str]:
    queries = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"queries file not found: {path}")
    current_name = None
    buffer = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if line.startswith("-- name:"):
            if current_name and buffer:
                queries[current_name] = "\n".join(buffer).strip()
                buffer = []
            current_name = line.split(":", 1)[1].strip()
            continue
        if line.strip().startswith("#") and not line.strip().upper().startswith("MATCH"):
            continue
        if current_name:
            buffer.append(line)
    if current_name and buffer:
        queries[current_name] = "\n".join(buffer).strip()
    return queries

# -------------------------
# Baseline retriever
# -------------------------

DEFAULT_LIMIT = 5

class BaselineRetriever:
    def __init__(self, config_path: str = None, queries_path: str = None):
        project_root = Path(__file__).resolve().parents[1]

        if config_path is None:
            config_path = str(project_root / "data" / "config.txt")
        if queries_path is None:
            queries_path = str(project_root / "data" / "queries.txt")

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found at {config_path}. Put your credentials in data/config.txt or pass --config.")
        if not Path(queries_path).exists():
            raise FileNotFoundError(f"Queries file not found at {queries_path}. Create it or pass --queries.")

        uri, user, password = load_config(config_path)
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.queries = load_queries(queries_path)

    def close(self):
        self.driver.close()

    def run_query(self, cypher: str, params: Dict[str, Any] = None):
        params = params or {}
        with self.driver.session() as session:
            result = session.run(cypher, params)
            return [record.data() for record in result]

    def retrieve(self, intent: str, entities: Dict[str, Any]):
        params = {}
        tpl = None

        if intent in ("recommendation", "hotel_search"):
            if entities.get("country") or entities.get("countries"):
                country_val = entities.get("country") or (entities.get("countries")[0] if entities.get("countries") else None)
                if country_val:
                    tpl = self.queries.get("hotels_by_country")
                    params["country"] = country_val
                    params["limit"] = entities.get("limit", DEFAULT_LIMIT)
            elif entities.get("city") or entities.get("cities"):
                city_val = entities.get("city") or (entities.get("cities")[0] if entities.get("cities") else None)
                tpl = self.queries.get("hotels_in_city")
                params["city"] = city_val
                params["limit"] = entities.get("limit", DEFAULT_LIMIT)
            elif entities.get("hotel_name") or entities.get("hotel_id") or entities.get("q"):
                tpl = self.queries.get("hotel_by_name")
                params["q"] = entities.get("hotel_name") or entities.get("hotel_id") or entities.get("q")
                params["limit"] = entities.get("limit", DEFAULT_LIMIT)
            else:
                tpl = self.queries.get("top_hotels")
                params["limit"] = entities.get("limit", DEFAULT_LIMIT)

        elif intent == "review_lookup":
            if entities.get("hotel_id"):
                tpl = self.queries.get("hotel_reviews")
                params["hotel_id"] = entities["hotel_id"]
                params["limit"] = entities.get("limit", DEFAULT_LIMIT)
            elif entities.get("hotel_name") or entities.get("q"):
                name_q = entities.get("hotel_name") or entities.get("q")
                name_tpl = self.queries.get("hotel_by_name")
                if name_tpl:
                    hits = self.run_query(name_tpl, {"q": name_q, "limit": 1})
                    if hits:
                        params["hotel_id"] = hits[0].get("hotel_id")
                        tpl = self.queries.get("hotel_reviews")
                        params["limit"] = entities.get("limit", DEFAULT_LIMIT)
                    else:
                        return []
                else:
                    return []
            else:
                return []

        elif intent == "hotels_with_min_reviews" or entities.get("min_reviews") is not None:
            tpl = self.queries.get("hotels_with_min_reviews")
            params["min_reviews"] = entities.get("min_reviews")
            params["limit"] = entities.get("limit", DEFAULT_LIMIT)

        elif intent in ("traveller_query", "traveller_type_preferences"):
            type_val = entities.get("type") or entities.get("traveler_type") or (entities.get("traveler_types")[0] if entities.get("traveler_types") else None)
            if not type_val:
                return []
            cypher = (
                "MATCH (t:Traveller)-[:STAYED_AT]->(h:Hotel) "
                "WHERE toLower(t.type) = toLower($type) "
                "WITH h, count(*) AS visits "
                "OPTIONAL MATCH (h)-[:LOCATED_IN]->(c:City) "
                "OPTIONAL MATCH (c)-[:LOCATED_IN]->(country:Country) "
                "RETURN h.hotel_id AS hotel_id, h.name AS name, visits, c.name AS city, country.name AS country "
                "ORDER BY visits DESC LIMIT $limit"
            )
            params["type"] = type_val
            params["limit"] = entities.get("limit", DEFAULT_LIMIT)
            return self.run_query(cypher, params)

        elif intent == "facility_search":
            if entities.get("min_facility_score") is not None:
                tpl = self.queries.get("hotels_by_avg_facilities_score")
                params["min_facility_score"] = entities.get("min_facility_score")
                params["limit"] = entities.get("limit", DEFAULT_LIMIT)
            elif entities.get("facility") or entities.get("facilities") or entities.get("facility_ratings"):
                tpl = self.queries.get("hotels_by_avg_facilities_score")
                min_score = None
                if entities.get("min_facility_score") is not None:
                    min_score = entities["min_facility_score"]
                elif entities.get("facility_ratings"):
                    vals = list(entities["facility_ratings"].values())
                    if vals:
                        min_score = max(vals)
                if min_score is None:
                    return []
                params["min_facility_score"] = min_score
                params["limit"] = entities.get("limit", DEFAULT_LIMIT)
            else:
                return []

        else:
            tpl = self.queries.get("hotel_by_name")
            params["q"] = entities.get("city") or entities.get("hotel_name") or ""
            params["limit"] = entities.get("limit", DEFAULT_LIMIT)

        if tpl is None:
            raise ValueError(f"No Cypher template found for chosen intent. Available templates: {list(self.queries.keys())}")

        return self.run_query(tpl, params)


# -------------------------
# Demo runner (quick test)
# -------------------------

def demo(config_path=None, queries_path=None):
    tests = [
        ("Hotels in city (Cairo)", "recommendation", {"city": "Cairo", "limit": 5}),
        ("Hotels in city with min rating (Cairo, >=4)", "recommendation", {"city": "Cairo", "min_rating": 4.0, "limit": 5}),
        ("Hotel by name (The Royal Compass)", "hotel_search", {"hotel_name": "The Royal Compass", "limit": 5}),
        ("Top hotels (by star_rating)", "hotel_search", {"limit": 5}),
        ("Hotels by country (Egypt)", "hotel_search", {"country": "Egypt", "limit": 5}),
        ("Most reviewed hotels in city (Cairo)", "hotel_search", {"city": "Cairo", "limit": 5}),
        ("Visa requirement Egypt -> France", "visa_query", {"country_from": "Egypt", "country_to": "France"}),
        ("Traveller type preferences (business)", "traveller_query", {"type": "business", "limit": 5}),
        ("Hotels with >= 100 reviews", "hotel_search", {"min_reviews": 100, "limit": 10, "intent_template": "hotels_with_min_reviews"}),
        ("Most active reviewers", "hotel_search", {"limit": 10, "intent_template": "most_active_reviewers"}),
    ]

    retriever = BaselineRetriever(config_path=config_path, queries_path=queries_path)
    try:
        for test_name, intent, entities in tests:
            print("\n" + "=" * 80)
            print("TEST:", test_name)
            print("Intent:", intent)
            print("Entities / params:", entities)
            try:
                results = retriever.retrieve(intent, entities)
            except Exception as e:
                print("Query failed with exception:", repr(e))
                continue

            print("Results count:", len(results))
            if not results:
                print("  <-- No results returned. Try the exploratory queries below to find valid city/hotel_id values.")
                continue

            # Print up to 3 sample rows
            for i, r in enumerate(results[:3], start=1):
                print(f" Sample {i}:", r)
    finally:
        retriever.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline retriever CLI (no client wrapper).")
    parser.add_argument('--demo', action='store_true', help='Run demo queries')
    parser.add_argument('--config', type=str, default=None, help='Path to config.txt (optional)')
    parser.add_argument('--queries', type=str, default=None, help='Path to queries.txt (optional)')
    args = parser.parse_args()
    if args.demo:
        demo(config_path=args.config, queries_path=args.queries)
    else:
        print("Run with --demo to execute sample queries against the Neo4j DB.")
