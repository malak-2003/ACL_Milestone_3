# dummy_input.py

dummy_retrieval_result = {
    "query": "Find good hotels in Cairo with a pool",
    "retrieval_method": "hybrid",
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
    "extra_context": [
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
}
