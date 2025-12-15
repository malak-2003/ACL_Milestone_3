import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

# Configuration
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  
    "mpnet": "sentence-transformers/all-mpnet-base-v2", 
}


def load_config(path="data/config.txt"):
    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    return config["URI"], config["USERNAME"], config["PASSWORD"]


class NodeEmbeddingGenerator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 model_name: str = "minilm"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.model_name = model_name
        
        print(f"Loading embedding model: {EMBEDDING_MODELS[model_name]}")
        self.model = SentenceTransformer(EMBEDDING_MODELS[model_name])
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
    def close(self):
        self.driver.close()
    
    def create_hotel_text_representation(self, hotel_data: Dict) -> str:
        parts = []
        
        city = hotel_data.get('city', '')
        country = hotel_data.get('country', '')
        
        if city and country:
            # Repeat location 3 times to give it strong weight
            parts.append(f"{city} {country}")
            parts.append(f"Located in {city}, {country}")
            parts.append(f"{city} hotel in {country}")
        
        # Hotel name
        if hotel_data.get('name'):
            parts.append(f"Hotel name: {hotel_data['name']}")
        
        # Star rating
        if hotel_data.get('star_rating'):
            rating = hotel_data['star_rating']
            if rating >= 5:
                parts.append(f"{rating}-star luxury hotel")
            elif rating >= 4:
                parts.append(f"{rating}-star upscale hotel")
            elif rating >= 3:
                parts.append(f"{rating}-star mid-range hotel")
            else:
                parts.append(f"{rating}-star budget hotel")
        
        # Quality scores
        if hotel_data.get('cleanliness_base'):
            score = hotel_data['cleanliness_base']
            if score >= 9:
                parts.append("Excellent cleanliness")
            elif score >= 8:
                parts.append("Very good cleanliness")
            elif score >= 7:
                parts.append("Good cleanliness")
        
        if hotel_data.get('facilities_base'):
            score = hotel_data['facilities_base']
            if score >= 9:
                parts.append("Excellent facilities")
            elif score >= 8:
                parts.append("Very good facilities")
        
        if hotel_data.get('comfort_base'):
            score = hotel_data['comfort_base']
            if score >= 9:
                parts.append("Excellent comfort")
            elif score >= 8:
                parts.append("Very good comfort")
        
        # Combine all parts
        text = " | ".join(str(p) for p in parts if p)
        return text
    
    def fetch_all_hotels(self) -> List[Dict]:
        query = """
        MATCH (h:Hotel)
        OPTIONAL MATCH (h)-[:LOCATED_IN]->(c:City)
        OPTIONAL MATCH (c)-[:LOCATED_IN]->(country:Country)
        RETURN h.hotel_id AS hotel_id,
               h.name AS name,
               h.star_rating AS star_rating,
               h.cleanliness_base AS cleanliness_base,
               h.comfort_base AS comfort_base,
               h.facilities_base AS facilities_base,
               c.name AS city,
               country.name AS country
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            hotels = [record.data() for record in result]
        
        print(f"Fetched {len(hotels)} hotels from Neo4j")
        return hotels
    
    def generate_embeddings(self, hotels: List[Dict]) -> Dict[str, np.ndarray]:
        embeddings = {}
        texts = []
        hotel_ids = []
        
        print("Creating text representations...")
        for hotel in tqdm(hotels):
            if hotel.get('hotel_id'):
                text = self.create_hotel_text_representation(hotel)
                texts.append(text)
                hotel_ids.append(hotel['hotel_id'])
        
        print(f"Generating embeddings for {len(texts)} hotels...")
        # Generate embeddings in batch for efficiency
        embedding_vectors = self.model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Map hotel_id to embedding
        for hotel_id, embedding in zip(hotel_ids, embedding_vectors):
            embeddings[hotel_id] = embedding
        
        return embeddings
    
    def create_vector_index(self, index_name: str = None):
        if index_name is None:
            index_name = f"hotel_embeddings_{self.model_name}"
        
        # Drop existing index if it exists
        drop_query = f"DROP INDEX {index_name} IF EXISTS"
        
        # Create vector index
        create_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (h:Hotel)
        ON h.embedding_{self.model_name}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        
        with self.driver.session() as session:
            try:
                session.run(drop_query)
                print(f"Dropped existing index: {index_name}")
            except Exception:
                pass
            
            try:
                session.run(create_query)
                print(f"Created vector index: {index_name}")
            except Exception as e:
                print(f"Note: {e}")
                print("Vector index may already exist or Neo4j version may not support vector indexes.")
    
    def store_embeddings_in_neo4j(self, embeddings: Dict[str, np.ndarray]):
        property_name = f"embedding_{self.model_name}"
        
        query = f"""
        MATCH (h:Hotel {{hotel_id: $hotel_id}})
        SET h.{property_name} = $embedding
        """
        
        print(f"Storing embeddings in Neo4j as 'h.{property_name}'...")
        with self.driver.session() as session:
            for hotel_id, embedding in tqdm(embeddings.items()):
                session.run(query, {
                    "hotel_id": hotel_id,
                    "embedding": np.array(embedding, dtype=np.float32).tolist()
                })
        
        print(f"Stored {len(embeddings)} embeddings in Neo4j")
    
    def save_embeddings_to_file(self, embeddings: Dict[str, np.ndarray], 
                                filepath: str = None):
        if filepath is None:
            filepath = f"embeddings_{self.model_name}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"Saved embeddings to {filepath}")
    
    def load_embeddings_from_file(self, filepath: str) -> Dict[str, np.ndarray]:
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        
        print(f"Loaded {len(embeddings)} embeddings from {filepath}")
        return embeddings
    
    def generate_and_store_all(self, save_to_file: bool = True):
        # Fetch hotels
        hotels = self.fetch_all_hotels()
        
        if not hotels:
            print("No hotels found in database!")
            return
        
        # Generate embeddings
        embeddings = self.generate_embeddings(hotels)
        
        # Store in Neo4j
        self.store_embeddings_in_neo4j(embeddings)
        
        # Create vector index (if supported)
        try:
            self.create_vector_index()
        except Exception as e:
            print(f"Could not create vector index: {e}")
        
        # Save to file
        if save_to_file:
            self.save_embeddings_to_file(embeddings)
        
        print(f"\n✓ Successfully generated and stored {len(embeddings)} embeddings using {self.model_name}")


class EmbeddingRetriever:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 model_name: str = "minilm"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.model_name = model_name
        
        print(f"Loading embedding model: {EMBEDDING_MODELS[model_name]}")
        self.model = SentenceTransformer(EMBEDDING_MODELS[model_name])
    
    def close(self):
        self.driver.close()
    
    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(query, convert_to_numpy=True)
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def find_similar_hotels_vector_index(self, query_embedding: np.ndarray, 
                                        limit: int = 10) -> List[Dict]:
        index_name = f"hotel_embeddings_{self.model_name}"
        
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $limit, $query_embedding)
        YIELD node, score
        MATCH (node)-[:LOCATED_IN]->(c:City)
        OPTIONAL MATCH (c)-[:LOCATED_IN]->(country:Country)
        RETURN node.hotel_id AS hotel_id,
               node.name AS name,
               node.star_rating AS star_rating,
               node.cleanliness_base AS cleanliness_base,
               c.name AS city,
               country.name AS country,
               score
        ORDER BY score DESC
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(query, {
                    "index_name": index_name,
                    "limit": limit,
                    "query_embedding": query_embedding.tolist()
                })
                return [record.data() for record in result]
            except Exception as e:
                print(f"Vector index query failed: {e}")
                print("Falling back to manual similarity search...")
                return self.find_similar_hotels_manual(query_embedding, limit)
    
    def find_similar_hotels_manual(self, query_embedding: np.ndarray,
                                   limit: int = 10, city_filter: str = None) -> List[Dict]:
        property_name = f"embedding_{self.model_name}"
        
        # Fetch all hotels with embeddings
        if city_filter:
            query = f"""
            MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
            WHERE h.{property_name} IS NOT NULL AND toLower(c.name) = toLower($city)
            OPTIONAL MATCH (c)-[:LOCATED_IN]->(country:Country)
            RETURN h.hotel_id AS hotel_id,
                   h.name AS name,
                   h.star_rating AS star_rating,
                   h.cleanliness_base AS cleanliness_base,
                   c.name AS city,
                   country.name AS country,
                   h.{property_name} AS embedding
            """
        else:
            query = f"""
            MATCH (h:Hotel)
            WHERE h.{property_name} IS NOT NULL
            OPTIONAL MATCH (h)-[:LOCATED_IN]->(c:City)
            OPTIONAL MATCH (c)-[:LOCATED_IN]->(country:Country)
            RETURN h.hotel_id AS hotel_id,
                   h.name AS name,
                   h.star_rating AS star_rating,
                   h.cleanliness_base AS cleanliness_base,
                   c.name AS city,
                   country.name AS country,
                   h.{property_name} AS embedding
            """
        
        with self.driver.session() as session:
            if city_filter:
                result = session.run(query, {"city": city_filter})
            else:
                result = session.run(query)
            hotels = [record.data() for record in result]
        
        if not hotels:
            print(f"Warning: No hotels found" + (f" in {city_filter}" if city_filter else ""))
            return []
        
        # Calculate cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        
        for hotel in hotels:
            hotel_embedding = np.array(hotel['embedding'])
            hotel_norm = np.linalg.norm(hotel_embedding)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, hotel_embedding) / (query_norm * hotel_norm)
            hotel['score'] = float(similarity)
            del hotel['embedding']  # Remove embedding from result
        
        # Sort by similarity and return top results
        hotels.sort(key=lambda x: x['score'], reverse=True)
        return hotels[:limit]
    
    def search(self, query: str, limit: int = 10, 
               use_vector_index: bool = True, city_filter: str = None,
               auto_detect_city: bool = True) -> List[Dict]:
        
        print(f"\nSearching with query: '{query}'")
        print(f"Using model: {self.model_name}")
        
        # Auto-detect city from query if enabled
        detected_city = None
        if auto_detect_city and not city_filter:
            # Simple city detection from common city names
            common_cities = [
                "Cairo", "Dubai", "Paris", "London", "New York", "Tokyo", 
                "Singapore", "Sydney", "Barcelona", "Rome", "Berlin",
                "Amsterdam", "Madrid", "Istanbul", "Bangkok", "Hong Kong",
                "Rio de Janeiro", "Buenos Aires", "Cape Town", "Seoul"
            ]
            query_lower = query.lower()
            for city in common_cities:
                if city.lower() in query_lower:
                    detected_city = city
                    print(f"Auto-detected city: {city}")
                    break
        
        final_city_filter = city_filter or detected_city
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search with city filter if available
        if final_city_filter:
            print(f"Filtering results to city: {final_city_filter}")
            results = self.find_similar_hotels_manual(query_embedding, limit, final_city_filter)
        else:
            if use_vector_index:
                results = self.find_similar_hotels_vector_index(query_embedding, limit)
            else:
                results = self.find_similar_hotels_manual(query_embedding, limit)
        
        print(f"Found {len(results)} results")
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate node embeddings for hotels")
    parser.add_argument('--config', type=str, default='data/config.txt',
                       help='Path to Neo4j config file')
    parser.add_argument('--model', type=str, choices=['minilm', 'mpnet', 'both'],
                       default='both', help='Which embedding model to use')
    parser.add_argument('--mode', type=str, choices=['generate', 'search'],
                       default='generate', help='Mode: generate embeddings or search')
    parser.add_argument('--query', type=str, default=None,
                       help='Search query (for search mode)')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of results (for search mode)')
    parser.add_argument('--city', type=str, default=None,
                       help='Filter results by city (optional)')
    parser.add_argument('--no-auto-city', action='store_true',
                       help='Disable automatic city detection from query')
    
    args = parser.parse_args()
    
    # Load config
    uri, user, password = load_config(args.config)
    
    if args.mode == 'generate':
        # Generate embeddings
        models_to_use = ['minilm', 'mpnet'] if args.model == 'both' else [args.model]
        
        for model_name in models_to_use:
            print(f"\n{'='*80}")
            print(f"Generating embeddings with model: {model_name}")
            print(f"{'='*80}\n")
            
            generator = NodeEmbeddingGenerator(uri, user, password, model_name)
            try:
                generator.generate_and_store_all(save_to_file=True)
            finally:
                generator.close()
            
            print(f"\n✓ Completed {model_name}\n")
    
    elif args.mode == 'search':
        if not args.query:
            print("Please provide a --query for search mode")
            return
        
        models_to_use = ['minilm', 'mpnet'] if args.model == 'both' else [args.model]
        
        for model_name in models_to_use:
            print(f"\n{'='*80}")
            print(f"Searching with model: {model_name}")
            print(f"{'='*80}")
            
            retriever = EmbeddingRetriever(uri, user, password, model_name)
            try:
                results = retriever.search(
                    args.query, 
                    limit=args.limit,
                    city_filter=args.city,
                    auto_detect_city=not args.no_auto_city
                )
                
                print(f"\nTop {len(results)} results:")
                for i, hotel in enumerate(results, 1):
                    print(f"\n{i}. {hotel['name']}")
                    print(f"   Location: {hotel.get('city', 'N/A')}, {hotel.get('country', 'N/A')}")
                    print(f"   Rating: {hotel.get('star_rating', 'N/A')}★")
                    print(f"   Similarity Score: {hotel.get('score', 0):.4f}")
            finally:
                retriever.close()


if __name__ == "__main__":
    main()