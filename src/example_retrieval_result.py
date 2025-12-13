# task2_output_example.py
# This shows the expected output format from Task 2 (Graph Retrieval Layer)
# Your Task 2 code should produce this structure

example_retrieval_result = {
    "query": "Find good hotels in Cairo with a pool",
    
    # Results from baseline Cypher queries
    "baseline_results": {
        "cypher_query": "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: 'Cairo'}) WHERE 'pool' IN h.amenities RETURN h",
        "execution_time": 0.045,
        "nodes": [
            {
                "type": "Hotel",
                "id": "hotel_1",
                "name": "Cairo Horizon Hotel",
                "city": "Cairo",
                "country": "Egypt",
                "rating": 4.6,
                "amenities": ["wifi", "pool", "breakfast"],
                "description": "A modern hotel with Nile views."
            },
            {
                "type": "Hotel",
                "id": "hotel_2",
                "name": "Nile Pearl Inn",
                "city": "Cairo",
                "country": "Egypt",
                "rating": 4.3,
                "amenities": ["wifi", "pool"],
                "description": "Affordable hotel near downtown Cairo."
            }
        ],
        "relationships": [
            {
                "type": "LOCATED_IN",
                "start": "hotel_1",
                "end": "Cairo"
            },
            {
                "type": "LOCATED_IN",
                "start": "hotel_2",
                "end": "Cairo"
            }
        ],
        "reviews": [
            {
                "type": "Review",
                "hotel_id": "hotel_1",
                "text": "Amazing view and very clean rooms.",
                "score": 5
            },
            {
                "type": "Review",
                "hotel_id": "hotel_2",
                "text": "Good value for money but breakfast is limited.",
                "score": 4
            }
        ]
    },
    
    # Results from embedding-based retrieval
    "embedding_results": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_embedding_time": 0.012,
        "search_time": 0.028,
        "similar_nodes": [
            {
                "type": "Hotel",
                "id": "hotel_1",  # Same as baseline - will be deduplicated
                "name": "Cairo Horizon Hotel",
                "city": "Cairo",
                "country": "Egypt",
                "rating": 4.6,
                "amenities": ["wifi", "pool", "breakfast"],
                "description": "A modern hotel with Nile views.",
                "similarity_score": 0.92  # High similarity to query
            },
            {
                "type": "Hotel",
                "id": "hotel_3",  # New hotel not found by baseline
                "name": "Pyramids View Resort",
                "city": "Cairo",
                "country": "Egypt",
                "rating": 4.5,
                "amenities": ["pool", "spa", "wifi"],
                "description": "Luxury resort with pyramid views and excellent pool facilities.",
                "similarity_score": 0.88
            }
        ],
        "reviews": []  # Reviews already in baseline
    }
}


# Alternative: If your Task 2 only implements baseline OR embeddings (not both yet)
example_baseline_only = {
    "query": "Find good hotels in Cairo with a pool",
    "baseline_results": {
        # ... same as above
    },
    "embedding_results": None  # Not implemented yet
}

example_embeddings_only = {
    "query": "Find good hotels in Cairo with a pool",
    "baseline_results": None,
    "embedding_results": {
        # ... same as above
    }
}