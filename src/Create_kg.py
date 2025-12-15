# kg.py

import csv
from neo4j import GraphDatabase

CONFIG_PATH = "../data/config.txt"

# ---------- 1. Read config ----------

def load_config(path=CONFIG_PATH):
    """
    Reads URI, USERNAME, PASSWORD from config.txt
    """
    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config["URI"], config["USERNAME"], config["PASSWORD"]

# ---------- 2. Helper converters ----------

def to_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None

def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_bool(value):
    if value is None:
        return False
    v = str(value).strip().lower()
    return v in ("1", "true", "yes", "y")


# ---------- 3. Graph Builder ----------

class HotelKGBuilder:
    """
    Builds the hotel Knowledge Graph exactly as required:
      Traveller, Hotel, City, Country, Review nodes
      and all relationships between them.
    """

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # ----- Setup -----

    def clear_database(self):
        """
        Delete all nodes & relationships (useful for development).
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        """
        Create uniqueness constraints on IDs/names as required.
        """
        queries = [
            "CREATE CONSTRAINT traveller_id IF NOT EXISTS FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE",
            "CREATE CONSTRAINT hotel_id IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE",
            "CREATE CONSTRAINT city_name IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT country_name IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT review_id IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    # ----- Import from hotels.csv -----

    def import_hotels_and_locations(self, hotels_csv="../data/hotels.csv"):
        """
        From hotels.csv:
          - Create Country(name)
          - Create City(name) and (City)-[:LOCATED_IN]->(Country)
          - Create Hotel(...) and (Hotel)-[:LOCATED_IN]->(City)
        """
        with self.driver.session() as session, open(hotels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hotel_id = row["hotel_id"]
                hotel_name = row["hotel_name"]
                city_name = row["city"]
                country_name = row["country"]

                star_rating = to_int(row.get("star_rating"))
                cleanliness_base = to_float(row.get("cleanliness_base"))
                comfort_base = to_float(row.get("comfort_base"))
                facilities_base = to_float(row.get("facilities_base"))

                # Country(name)
                session.run(
                    """
                    MERGE (country:Country {name: $country_name})
                    """,
                    country_name=country_name,
                )

                # City(name) and City-LOCATED_IN->Country
                session.run(
                    """
                    MERGE (country:Country {name: $country_name})
                    MERGE (city:City {name: $city_name})
                    MERGE (city)-[:LOCATED_IN]->(country)
                    """,
                    city_name=city_name,
                    country_name=country_name,
                )

                # Hotel node with required properties
                session.run(
                    """
                    MERGE (h:Hotel {hotel_id: $hotel_id})
                    SET h.name = $name,
                        h.star_rating = $star_rating,
                        h.cleanliness_base = $cleanliness_base,
                        h.comfort_base = $comfort_base,
                        h.facilities_base = $facilities_base
                    WITH h
                    MATCH (c:City {name: $city_name})
                    MERGE (h)-[:LOCATED_IN]->(c)
                    """,
                    hotel_id=hotel_id,
                    name=hotel_name,
                    star_rating=star_rating,
                    cleanliness_base=cleanliness_base,
                    comfort_base=comfort_base,
                    facilities_base=facilities_base,
                    city_name=city_name,
                )

    # ----- Import from users.csv -----

    def import_travellers(self, users_csv="../data/users.csv"):
        """
        From users.csv:
          - Create Traveller(user_id, age, type, gender)
          - Create (Traveller)-[:FROM_COUNTRY]->(Country)
        """
        with self.driver.session() as session, open(users_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row["user_id"]
                age = row.get("age_group")
                gender = row.get("user_gender")
                traveller_type = row.get("traveller_type")
                country_name = row.get("country")

                # Traveller node with required properties
                session.run(
                    """
                    MERGE (t:Traveller {user_id: $user_id})
                    SET t.age = $age,
                        t.gender = $gender,
                        t.type = $type
                    """,
                    user_id=user_id,
                    age=age,
                    gender=gender,
                    type=traveller_type,
                )

                # Traveller-[:FROM_COUNTRY]->Country
                if country_name:
                    session.run(
                        """
                        MERGE (c:Country {name: $country_name})
                        WITH c
                        MATCH (t:Traveller {user_id: $user_id})
                        MERGE (t)-[:FROM_COUNTRY]->(c)
                        """,
                        user_id=user_id,
                        country_name=country_name,
                    )

    # ----- Import from reviews.csv -----

    def import_reviews_and_stays(self, reviews_csv="../data/reviews.csv"):
        """
        From reviews.csv:
          - Create Review node with all required score properties
          - Link Traveller and Hotel:
              (Traveller)-[:WROTE]->(Review)
              (Review)-[:REVIEWED]->(Hotel)
              (Traveller)-[:STAYED_AT]->(Hotel)
        """
        with self.driver.session() as session, open(reviews_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                review_id = row["review_id"]
                user_id = row["user_id"]
                hotel_id = row["hotel_id"]

                text = row.get("text")
                date = row.get("date")

                score_overall = to_float(row.get("score_overall"))
                score_cleanliness = to_float(row.get("score_cleanliness"))
                score_comfort = to_float(row.get("score_comfort"))
                score_facilities = to_float(row.get("score_facilities"))
                score_location = to_float(row.get("score_location"))
                score_staff = to_float(row.get("score_staff"))
                score_value_for_money = to_float(row.get("score_value_for_money"))

                # Review node
                session.run(
                    """
                    MERGE (r:Review {review_id: $review_id})
                    SET r.text = $text,
                        r.date = $date,
                        r.score_overall = $score_overall,
                        r.score_cleanliness = $score_cleanliness,
                        r.score_comfort = $score_comfort,
                        r.score_facilities = $score_facilities,
                        r.score_location = $score_location,
                        r.score_staff = $score_staff,
                        r.score_value_for_money = $score_value_for_money
                    """,
                    review_id=review_id,
                    text=text,
                    date=date,
                    score_overall=score_overall,
                    score_cleanliness=score_cleanliness,
                    score_comfort=score_comfort,
                    score_facilities=score_facilities,
                    score_location=score_location,
                    score_staff=score_staff,
                    score_value_for_money=score_value_for_money,
                )

                # Relationships:
                # (Traveller)-[:WROTE]->(Review)
                # (Review)-[:REVIEWED]->(Hotel)
                # (Traveller)-[:STAYED_AT]->(Hotel)
                session.run(
                    """
                    MATCH (t:Traveller {user_id: $user_id})
                    MATCH (h:Hotel {hotel_id: $hotel_id})
                    MATCH (r:Review {review_id: $review_id})
                    MERGE (t)-[:WROTE]->(r)
                    MERGE (r)-[:REVIEWED]->(h)
                    MERGE (t)-[:STAYED_AT]->(h)
                    """,
                    user_id=user_id,
                    hotel_id=hotel_id,
                    review_id=review_id,
                )

    # ----- Import from visa.csv -----

    def import_visas(self, visa_csv="../data/visa.csv"):
        """
        From visa.csv:
          - For rows where requires = true, create:
              (Country {from})-[:NEEDS_VISA {visa_type}]->(Country {to})
        """
        with self.driver.session() as session, open(visa_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                from_country = row["from"]
                to_country = row["to"]
                requires = to_bool(row.get("requires_visa"))
                visa_type = row.get("visa_type")

                if not requires:
                    continue

                session.run(
                    """
                    MERGE (fromC:Country {name: $from_country})
                    MERGE (toC:Country {name: $to_country})
                    MERGE (fromC)-[r:NEEDS_VISA]->(toC)
                    SET r.visa_type = $visa_type
                    """,
                    from_country=from_country,
                    to_country=to_country,
                    visa_type=visa_type,
                )

    # ----- Full pipeline -----

    def build_graph(self):
        """
        Build the full KG from CSVs in the correct schema.
        """
        print("Clearing database...")
        self.clear_database()

        print("Creating constraints...")
        self.create_constraints()

        print("Importing hotels and locations...")
        self.import_hotels_and_locations()

        print("Importing travellers...")
        self.import_travellers()

        print("Importing reviews and stays...")
        self.import_reviews_and_stays()

        print("Importing visa information...")
        self.import_visas()

        print("Knowledge graph created successfully.")


# ---------- Entry point ----------

if __name__ == "__main__":
    uri, user, password = load_config()
    builder = HotelKGBuilder(uri, user, password)
    try:
        builder.build_graph()
    finally:
        builder.close()
