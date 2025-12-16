# task2_output_example.py
# This shows the expected output format from Task 2 (Graph Retrieval Layer)
# Your Task 2 code should produce this structure

# example_retrieval_result = {
#     "query": "Find good hotels in Cairo with a pool",
    
#     # Results from baseline Cypher queries
#     "baseline_results": {
#         "cypher_query": "MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: 'Cairo'}) WHERE 'pool' IN h.amenities RETURN h",
#         "execution_time": 0.045,
#         "nodes": [
#             {
#                 "type": "Hotel",
#                 "id": "hotel_1",
#                 "name": "Cairo Horizon Hotel",
#                 "city": "Cairo",
#                 "country": "Egypt",
#                 "rating": 4.6,
#                 "amenities": ["wifi", "pool", "breakfast"],
#                 "description": "A modern hotel with Nile views."
#             },
#             {
#                 "type": "Hotel",
#                 "id": "hotel_2",
#                 "name": "Nile Pearl Inn",
#                 "city": "Cairo",
#                 "country": "Egypt",
#                 "rating": 4.3,
#                 "amenities": ["wifi", "pool"],
#                 "description": "Affordable hotel near downtown Cairo."
#             }
#         ],
#         "relationships": [
#             {
#                 "type": "LOCATED_IN",
#                 "start": "hotel_1",
#                 "end": "Cairo"
#             },
#             {
#                 "type": "LOCATED_IN",
#                 "start": "hotel_2",
#                 "end": "Cairo"
#             }
#         ],
#         "reviews": [
#             {
#                 "type": "Review",
#                 "hotel_id": "hotel_1",
#                 "text": "Amazing view and very clean rooms.",
#                 "score": 5
#             },
#             {
#                 "type": "Review",
#                 "hotel_id": "hotel_2",
#                 "text": "Good value for money but breakfast is limited.",
#                 "score": 4
#             }
#         ]
#     },
    
#     # Results from embedding-based retrieval
#     "embedding_results": {
#         "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
#         "query_embedding_time": 0.012,
#         "search_time": 0.028,
#         "similar_nodes": [
#             {
#                 "type": "Hotel",
#                 "id": "hotel_1",  # Same as baseline - will be deduplicated
#                 "name": "Cairo Horizon Hotel",
#                 "city": "Cairo",
#                 "country": "Egypt",
#                 "rating": 4.6,
#                 "amenities": ["wifi", "pool", "breakfast"],
#                 "description": "A modern hotel with Nile views.",
#                 "similarity_score": 0.92  # High similarity to query
#             },
#             {
#                 "type": "Hotel",
#                 "id": "hotel_3",  # New hotel not found by baseline
#                 "name": "Pyramids View Resort",
#                 "city": "Cairo",
#                 "country": "Egypt",
#                 "rating": 4.5,
#                 "amenities": ["pool", "spa", "wifi"],
#                 "description": "Luxury resort with pyramid views and excellent pool facilities.",
#                 "similarity_score": 0.88
#             }
#         ],
#         "reviews": []  # Reviews already in baseline
#     }
# }

# example_retrieval_result = {
#     "query": "Show me The Royal Compass",
#     "results": {
#         "baseline": {
#             "nodes": [
#                 {
#                     "hotel_id": "2",
#                     "name": "The Royal Compass",
#                     "city": "London",
#                     "country": "United Kingdom",
#                     "star_rating": 5,
#                     "cleanliness": None
#                 }
#             ],
#             "reviews": [
#                 {
#                     "review_id": "16168",
#                     "review_text": "Front start executive spring only.\nShare short down anything oil Mrs challenge. Important specific operation book cost according itself.",
#                     "review_date": "2025-12-30",
#                     "score_overall": 8.9,
#                     "score_cleanliness": 9.5,
#                     "score_comfort": 9.3,
#                     "score_facilities": 8.2,
#                     "score_location": 9.2,
#                     "score_staff": 9.0,
#                     "score_value_for_money": 8.0
#                 },
#                 {
#                     "review_id": "40404",
#                     "review_text": "Stuff power front late hard position property. Pick force bill police interview technology firm. Seem they college between hold.",
#                     "review_date": "2025-12-29",
#                     "score_overall": 9.1,
#                     "score_cleanliness": 9.1,
#                     "score_comfort": 9.1,
#                     "score_facilities": 8.8,
#                     "score_location": 9.2,
#                     "score_staff": 9.5,
#                     "score_value_for_money": 8.9
#                 },
#                 {
#                     "review_id": "12758",
#                     "review_text": "Standard while feeling marriage. Cover protect gas imagine them generation.",
#                     "review_date": "2025-12-27",
#                     "score_overall": 9.0,
#                     "score_cleanliness": 9.5,
#                     "score_comfort": 8.7,
#                     "score_facilities": 9.5,
#                     "score_location": 9.1,
#                     "score_staff": 9.5,
#                     "score_value_for_money": 7.8
#                 }
#             ]
#         },
#         "minilm": {
#             "nodes": [
#                 {
#                     "hotel_id": "2",
#                     "name": "The Royal Compass",
#                     "city": "London",
#                     "country": "United Kingdom",
#                     "star_rating": 5,
#                     "cleanliness": 9.0,
#                     "similarity_score": 0.6751792430877686
#                 }
#             ],
#             "reviews": [
#                 {
#                     "review_id": "16168",
#                     "review_text": "Front start executive spring only.\nShare short down anything oil Mrs challenge. Important specific operation book cost according itself.",
#                     "review_date": "2025-12-30",
#                     "score_overall": 8.9,
#                     "score_cleanliness": 9.5,
#                     "score_comfort": 9.3,
#                     "score_facilities": 8.2,
#                     "score_location": 9.2,
#                     "score_staff": 9.0,
#                     "score_value_for_money": 8.0
#                 },
#                 {
#                     "review_id": "40404",
#                     "review_text": "Stuff power front late hard position property. Pick force bill police interview technology firm. Seem they college between hold.",
#                     "review_date": "2025-12-29",
#                     "score_overall": 9.1,
#                     "score_cleanliness": 9.1,
#                     "score_comfort": 9.1,
#                     "score_facilities": 8.8,
#                     "score_location": 9.2,
#                     "score_staff": 9.5,
#                     "score_value_for_money": 8.9
#                 },
#                 {
#                     "review_id": "12758",
#                     "review_text": "Standard while feeling marriage. Cover protect gas imagine them generation.",
#                     "review_date": "2025-12-27",
#                     "score_overall": 9.0,
#                     "score_cleanliness": 9.5,
#                     "score_comfort": 8.7,
#                     "score_facilities": 9.5,
#                     "score_location": 9.1,
#                     "score_staff": 9.5,
#                     "score_value_for_money": 7.8
#                 }
#             ]
#         },
#         "mpnet": {
#             "nodes": [
#                 {
#                     "hotel_id": "13",
#                     "name": "The Gateway Royale",
#                     "city": "Mumbai",
#                     "country": "India",
#                     "star_rating": 5,
#                     "cleanliness": 8.9,
#                     "similarity_score": 0.5931969285011292
#                 }
#             ],
#             "reviews": [
#                 {
#                     "review_id": "34316",
#                     "review_text": "Spring reality daughter six establish either need. Investment like support toward unit address hot beat. Care rise south myself star.",
#                     "review_date": "2025-12-31",
#                     "score_overall": 9.0,
#                     "score_cleanliness": 9.2,
#                     "score_comfort": 9.1,
#                     "score_facilities": 8.9,
#                     "score_location": 9.3,
#                     "score_staff": 8.7,
#                     "score_value_for_money": 8.8
#                 },
#                 {
#                     "review_id": "20271",
#                     "review_text": "Specific lead identify theory.\nAmerican instead position may response. Good feeling political that street such thank. Set senior many human here on.",
#                     "review_date": "2025-12-29",
#                     "score_overall": 9.0,
#                     "score_cleanliness": 9.4,
#                     "score_comfort": 9.5,
#                     "score_facilities": 8.0,
#                     "score_location": 9.4,
#                     "score_staff": 8.6,
#                     "score_value_for_money": 8.8
#                 },
#                 {
#                     "review_id": "20757",
#                     "review_text": "Turn close executive certainly heavy vote. Office ago tree seven.\nArrive something head wait clearly. His whom section my knowledge get simple.",
#                     "review_date": "2025-12-29",
#                     "score_overall": 8.9,
#                     "score_cleanliness": 9.1,
#                     "score_comfort": 8.1,
#                     "score_facilities": 8.9,
#                     "score_location": 9.4,
#                     "score_staff": 9.2,
#                     "score_value_for_money": 8.7
#                 }
#             ]
#         }
#     }
# }


# example_retrieval_result = {



#   "query": "Show me The Royal Compass",
#   "results": {
#     "baseline": {
#       "nodes": [
#         {
#           "hotel_id": "2",
#           "name": "The Royal Compass",
#           "city": "London",
#           "country": "United Kingdom",
#           "star_rating": 5,
#           "cleanliness": null
#         }
#       ],
#       "reviews": [
#         {
#           "review_id": "16168",
#           "review_text": "Front start executive spring only.\nShare short down anything oil Mrs challenge. Important specific operation book cost according itself.",
#           "review_date": "2025-12-30",
#           "score_overall": 8.9,
#           "score_cleanliness": 9.5,
#           "score_comfort": 9.3,
#           "score_facilities": 8.2,
#           "score_location": 9.2,
#           "score_staff": 9.0,
#           "score_value_for_money": 8.0
#         },
#         {
#           "review_id": "40404",
#           "review_text": "Stuff power front late hard position property. Pick force bill police interview technology firm. Seem they college between hold.",
#           "review_date": "2025-12-29",
#           "score_overall": 9.1,
#           "score_cleanliness": 9.1,
#           "score_comfort": 9.1,
#           "score_facilities": 8.8,
#           "score_location": 9.2,
#           "score_staff": 9.5,
#           "score_value_for_money": 8.9
#         },
#         {
#           "review_id": "12758",
#           "review_text": "Standard while feeling marriage. Cover protect gas imagine them generation.",
#           "review_date": "2025-12-27",
#           "score_overall": 9.0,
#           "score_cleanliness": 9.5,
#           "score_comfort": 8.7,
#           "score_facilities": 9.5,
#           "score_location": 9.1,
#           "score_staff": 9.5,
#           "score_value_for_money": 7.8
#         }
#       ]
#     },
#     "minilm": {
#       "nodes": [
#         {
#           "hotel_id": "2",
#           "name": "The Royal Compass",
#           "city": "London",
#           "country": "United Kingdom",
#           "star_rating": 5,
#           "cleanliness": 9.0,
#           "similarity_score": 0.6751792430877686
#         }
#       ],
#       "reviews": [
#         {
#           "review_id": "16168",
#           "review_text": "Front start executive spring only.\nShare short down anything oil Mrs challenge. Important specific operation book cost according itself.",
#           "review_date": "2025-12-30",
#           "score_overall": 8.9,
#           "score_cleanliness": 9.5,
#           "score_comfort": 9.3,
#           "score_facilities": 8.2,
#           "score_location": 9.2,
#           "score_staff": 9.0,
#           "score_value_for_money": 8.0
#         },
#         {
#           "review_id": "40404",
#           "review_text": "Stuff power front late hard position property. Pick force bill police interview technology firm. Seem they college between hold.",
#           "review_date": "2025-12-29",
#           "score_overall": 9.1,
#           "score_cleanliness": 9.1,
#           "score_comfort": 9.1,
#           "score_facilities": 8.8,
#           "score_location": 9.2,
#           "score_staff": 9.5,
#           "score_value_for_money": 8.9
#         },
#         {
#           "review_id": "12758",
#           "review_text": "Standard while feeling marriage. Cover protect gas imagine them generation.",
#           "review_date": "2025-12-27",
#           "score_overall": 9.0,
#           "score_cleanliness": 9.5,
#           "score_comfort": 8.7,
#           "score_facilities": 9.5,
#           "score_location": 9.1,
#           "score_staff": 9.5,
#           "score_value_for_money": 7.8
#         }
#       ]
#     },
#     "mpnet": {
#       "nodes": [
#         {
#           "hotel_id": "13",
#           "name": "The Gateway Royale",
#           "city": "Mumbai",
#           "country": "India",
#           "star_rating": 5,
#           "cleanliness": 8.9,
#           "similarity_score": 0.5931969285011292
#         }
#       ],
#       "reviews": [
#         {
#           "review_id": "34316",
#           "review_text": "Spring reality daughter six establish either need. Investment like support toward unit address hot beat. Care rise south myself star.",
#           "review_date": "2025-12-31",
#           "score_overall": 9.0,
#           "score_cleanliness": 9.2,
#           "score_comfort": 9.1,
#           "score_facilities": 8.9,
#           "score_location": 9.3,
#           "score_staff": 8.7,
#           "score_value_for_money": 8.8
#         },
#         {
#           "review_id": "20271",
#           "review_text": "Specific lead identify theory.\nAmerican instead position may response. Good feeling political that street such thank. Set senior many human here on.",
#           "review_date": "2025-12-29",
#           "score_overall": 9.0,
#           "score_cleanliness": 9.4,
#           "score_comfort": 9.5,
#           "score_facilities": 8.0,
#           "score_location": 9.4,
#           "score_staff": 8.6,
#           "score_value_for_money": 8.8
#         },
#         {
#           "review_id": "20757",
#           "review_text": "Turn close executive certainly heavy vote. Office ago tree seven.\nArrive something head wait clearly. His whom section my knowledge get simple.",
#           "review_date": "2025-12-29",
#           "score_overall": 8.9,
#           "score_cleanliness": 9.1,
#           "score_comfort": 8.1,
#           "score_facilities": 8.9,
#           "score_location": 9.4,
#           "score_staff": 9.2,
#           "score_value_for_money": 8.7
#         }
#       ]
#     }
#   }

# }



example_retrieval_result = {
  "query": "Show me The Royal Compass",
  "results": {
    "baseline": {
      "nodes": [
        {
          "hotel_id": "2",
          "name": "The Royal Compass",
          "city": "London",
          "country": "United Kingdom",
          "star_rating": 5,
          "reviews": [
            {
              "review_id": "16168",
              "review_text": "Front start executive spring only.\nShare short down anything oil Mrs challenge. Important specific operation book cost according itself.",
              "review_date": "2025-12-30",
              "score_overall": 8.9,
              "score_cleanliness": 9.5,
              "score_comfort": 9.3,
              "score_facilities": 8.2,
              "score_location": 9.2,
              "score_staff": 9.0,
              "score_value_for_money": 8.0
            },
            {
              "review_id": "40404",
              "review_text": "Stuff power front late hard position property. Pick force bill police interview technology firm. Seem they college between hold.",
              "review_date": "2025-12-29",
              "score_overall": 9.1,
              "score_cleanliness": 9.1,
              "score_comfort": 9.1,
              "score_facilities": 8.8,
              "score_location": 9.2,
              "score_staff": 9.5,
              "score_value_for_money": 8.9
            },
            {
              "review_id": "12758",
              "review_text": "Standard while feeling marriage. Cover protect gas imagine them generation.",
              "review_date": "2025-12-27",
              "score_overall": 9.0,
              "score_cleanliness": 9.5,
              "score_comfort": 8.7,
              "score_facilities": 9.5,
              "score_location": 9.1,
              "score_staff": 9.5,
              "score_value_for_money": 7.8
            }
          ]
        }
      ],
      "reviews": []
    },
    "minilm": {
      "nodes": [
        {
          "hotel_id": "2",
          "name": "The Royal Compass",
          "city": "London",
          "country": "United Kingdom",
          "star_rating": 5,
          "similarity_score": 0.6751793622970581,
          "reviews": [
            {
              "review_id": "16168",
              "review_text": "Front start executive spring only.\nShare short down anything oil Mrs challenge. Important specific operation book cost according itself.",
              "review_date": "2025-12-30",
              "score_overall": 8.9,
              "score_cleanliness": 9.5,
              "score_comfort": 9.3,
              "score_facilities": 8.2,
              "score_location": 9.2,
              "score_staff": 9.0,
              "score_value_for_money": 8.0
            },
            {
              "review_id": "40404",
              "review_text": "Stuff power front late hard position property. Pick force bill police interview technology firm. Seem they college between hold.",
              "review_date": "2025-12-29",
              "score_overall": 9.1,
              "score_cleanliness": 9.1,
              "score_comfort": 9.1,
              "score_facilities": 8.8,
              "score_location": 9.2,
              "score_staff": 9.5,
              "score_value_for_money": 8.9
            },
            {
              "review_id": "12758",
              "review_text": "Standard while feeling marriage. Cover protect gas imagine them generation.",
              "review_date": "2025-12-27",
              "score_overall": 9.0,
              "score_cleanliness": 9.5,
              "score_comfort": 8.7,
              "score_facilities": 9.5,
              "score_location": 9.1,
              "score_staff": 9.5,
              "score_value_for_money": 7.8
            }
          ],
          "cleanliness": 9.0
        },
        {
          "hotel_id": "24",
          "name": "The Savannah House",
          "city": "Lagos",
          "country": "Nigeria",
          "star_rating": 5,
          "similarity_score": 0.5696887373924255,
          "reviews": [
            {
              "review_id": "17829",
              "review_text": "Week on song form. Will issue large generation. The food beyond radio eye.",
              "review_date": "2025-12-30",
              "score_overall": 8.4,
              "score_cleanliness": 7.9,
              "score_comfort": 8.8,
              "score_facilities": 8.2,
              "score_location": 9.0,
              "score_staff": 8.5,
              "score_value_for_money": 7.8
            },
            {
              "review_id": "27136",
              "review_text": "Off fight little cut thank unit improve. Oil expect trial certain new discuss.",
              "review_date": "2025-12-30",
              "score_overall": 8.8,
              "score_cleanliness": 9.1,
              "score_comfort": 8.5,
              "score_facilities": 9.1,
              "score_location": 8.8,
              "score_staff": 8.4,
              "score_value_for_money": 8.6
            },
            {
              "review_id": "26795",
              "review_text": "Admit catch after west. Big lawyer concern tend upon hand three. Scientist too street break.",
              "review_date": "2025-12-29",
              "score_overall": 8.7,
              "score_cleanliness": 9.1,
              "score_comfort": 8.9,
              "score_facilities": 8.3,
              "score_location": 9.4,
              "score_staff": 8.0,
              "score_value_for_money": 7.9
            }
          ],
          "cleanliness": 8.7
        },
        {
          "hotel_id": "15",
          "name": "Table Mountain View",
          "city": "Cape Town",
          "country": "South Africa",
          "star_rating": 5,
          "similarity_score": 0.5468575954437256,
          "reviews": [
            {
              "review_id": "12698",
              "review_text": "Suffer occur guess thus technology however must bag. Son senior instead believe.",
              "review_date": "2025-12-31",
              "score_overall": 9.1,
              "score_cleanliness": 9.1,
              "score_comfort": 9.2,
              "score_facilities": 9.5,
              "score_location": 9.0,
              "score_staff": 8.9,
              "score_value_for_money": 8.9
            },
            {
              "review_id": "25690",
              "review_text": "Likely close side its vote. Window role final important rest different. Good mother expect beautiful change.",
              "review_date": "2025-12-28",
              "score_overall": 8.9,
              "score_cleanliness": 8.8,
              "score_comfort": 9.8,
              "score_facilities": 8.0,
              "score_location": 9.5,
              "score_staff": 8.5,
              "score_value_for_money": 8.7
            },
            {
              "review_id": "14339",
              "review_text": "Likely sure do leg. Answer go total catch imagine education.\nBest cup thought deal bring want image. There compare give bill involve beat painting.",
              "review_date": "2025-12-27",
              "score_overall": 9.0,
              "score_cleanliness": 9.7,
              "score_comfort": 9.8,
              "score_facilities": 7.8,
              "score_location": 9.3,
              "score_staff": 8.4,
              "score_value_for_money": 8.5
            }
          ],
          "cleanliness": 9.0
        },
        {
          "hotel_id": "13",
          "name": "The Gateway Royale",
          "city": "Mumbai",
          "country": "India",
          "star_rating": 5,
          "similarity_score": 0.5233106017112732,
          "reviews": [
            {
              "review_id": "34316",
              "review_text": "Spring reality daughter six establish either need. Investment like support toward unit address hot beat. Care rise south myself star.",
              "review_date": "2025-12-31",
              "score_overall": 9.0,
              "score_cleanliness": 9.2,
              "score_comfort": 9.1,
              "score_facilities": 8.9,
              "score_location": 9.3,
              "score_staff": 8.7,
              "score_value_for_money": 8.8
            },
            {
              "review_id": "20271",
              "review_text": "Specific lead identify theory.\nAmerican instead position may response. Good feeling political that street such thank. Set senior many human here on.",
              "review_date": "2025-12-29",
              "score_overall": 9.0,
              "score_cleanliness": 9.4,
              "score_comfort": 9.5,
              "score_facilities": 8.0,
              "score_location": 9.4,
              "score_staff": 8.6,
              "score_value_for_money": 8.8
            },
            {
              "review_id": "20757",
              "review_text": "Turn close executive certainly heavy vote. Office ago tree seven.\nArrive something head wait clearly. His whom section my knowledge get simple.",
              "review_date": "2025-12-29",
              "score_overall": 8.9,
              "score_cleanliness": 9.1,
              "score_comfort": 8.1,
              "score_facilities": 8.9,
              "score_location": 9.4,
              "score_staff": 9.2,
              "score_value_for_money": 8.7
            }
          ],
          "cleanliness": 8.9
        },
        {
          "hotel_id": "8",
          "name": "Copacabana Lux",
          "city": "Rio de Janeiro",
          "country": "Brazil",
          "star_rating": 5,
          "similarity_score": 0.5226398706436157,
          "reviews": [
            {
              "review_id": "14785",
              "review_text": "Mean skill manager exist list like level. It very upon shoulder always pass conference. Season strong them beyond pass evening.",
              "review_date": "2025-12-31",
              "score_overall": 8.8,
              "score_cleanliness": 9.2,
              "score_comfort": 9.2,
              "score_facilities": 8.5,
              "score_location": 9.3,
              "score_staff": 8.3,
              "score_value_for_money": 8.1
            },
            {
              "review_id": "30851",
              "review_text": "Piece fast film also century. Star energy sense arrive defense because trouble.",
              "review_date": "2025-12-31",
              "score_overall": 9.2,
              "score_cleanliness": 9.4,
              "score_comfort": 9.0,
              "score_facilities": 9.3,
              "score_location": 9.4,
              "score_staff": 8.9,
              "score_value_for_money": 8.9
            },
            {
              "review_id": "46012",
              "review_text": "Role performance factor car try significant eight. Rather animal best economy thought. Several build result.",
              "review_date": "2025-12-29",
              "score_overall": 9.0,
              "score_cleanliness": 9.4,
              "score_comfort": 9.2,
              "score_facilities": 8.4,
              "score_location": 9.0,
              "score_staff": 8.9,
              "score_value_for_money": 8.8
            }
          ],
          "cleanliness": 9.0
        }
      ],
      "reviews": []
    },
    "mpnet": {
      "nodes": [
        {
          "hotel_id": "2",
          "name": "The Royal Compass",
          "city": "London",
          "country": "United Kingdom",
          "star_rating": 5,
          "similarity_score": 0.764286458492279,
          "reviews": [
            {
              "review_id": "16168",
              "review_text": "Front start executive spring only.\nShare short down anything oil Mrs challenge. Important specific operation book cost according itself.",
              "review_date": "2025-12-30",
              "score_overall": 8.9,
              "score_cleanliness": 9.5,
              "score_comfort": 9.3,
              "score_facilities": 8.2,
              "score_location": 9.2,
              "score_staff": 9.0,
              "score_value_for_money": 8.0
            },
            {
              "review_id": "40404",
              "review_text": "Stuff power front late hard position property. Pick force bill police interview technology firm. Seem they college between hold.",
              "review_date": "2025-12-29",
              "score_overall": 9.1,
              "score_cleanliness": 9.1,
              "score_comfort": 9.1,
              "score_facilities": 8.8,
              "score_location": 9.2,
              "score_staff": 9.5,
              "score_value_for_money": 8.9
            },
            {
              "review_id": "12758",
              "review_text": "Standard while feeling marriage. Cover protect gas imagine them generation.",
              "review_date": "2025-12-27",
              "score_overall": 9.0,
              "score_cleanliness": 9.5,
              "score_comfort": 8.7,
              "score_facilities": 9.5,
              "score_location": 9.1,
              "score_staff": 9.5,
              "score_value_for_money": 7.8
            }
          ],
          "cleanliness": 9.0
        },
        {
          "hotel_id": "13",
          "name": "The Gateway Royale",
          "city": "Mumbai",
          "country": "India",
          "star_rating": 5,
          "similarity_score": 0.5931969285011292,
          "reviews": [
            {
              "review_id": "34316",
              "review_text": "Spring reality daughter six establish either need. Investment like support toward unit address hot beat. Care rise south myself star.",
              "review_date": "2025-12-31",
              "score_overall": 9.0,
              "score_cleanliness": 9.2,
              "score_comfort": 9.1,
              "score_facilities": 8.9,
              "score_location": 9.3,
              "score_staff": 8.7,
              "score_value_for_money": 8.8
            },
            {
              "review_id": "20271",
              "review_text": "Specific lead identify theory.\nAmerican instead position may response. Good feeling political that street such thank. Set senior many human here on.",
              "review_date": "2025-12-29",
              "score_overall": 9.0,
              "score_cleanliness": 9.4,
              "score_comfort": 9.5,
              "score_facilities": 8.0,
              "score_location": 9.4,
              "score_staff": 8.6,
              "score_value_for_money": 8.8
            },
            {
              "review_id": "20757",
              "review_text": "Turn close executive certainly heavy vote. Office ago tree seven.\nArrive something head wait clearly. His whom section my knowledge get simple.",
              "review_date": "2025-12-29",
              "score_overall": 8.9,
              "score_cleanliness": 9.1,
              "score_comfort": 8.1,
              "score_facilities": 8.9,
              "score_location": 9.4,
              "score_staff": 9.2,
              "score_value_for_money": 8.7
            }
          ],
          "cleanliness": 8.9
        },
        {
          "hotel_id": "1",
          "name": "The Azure Tower",
          "city": "New York",
          "country": "United States",
          "star_rating": 5,
          "similarity_score": 0.591213047504425,
          "reviews": [
            {
              "review_id": "558",
              "review_text": "Pick nor Congress black movement. However national former art Mrs herself themselves. Arm consumer old close time tree.",
              "review_date": "2025-12-30",
              "score_overall": 8.8,
              "score_cleanliness": 9.4,
              "score_comfort": 9.1,
              "score_facilities": 9.1,
              "score_location": 8.7,
              "score_staff": 8.3,
              "score_value_for_money": 7.8
            },
            {
              "review_id": "32207",
              "review_text": "Huge authority plan positive campaign American key. Building stay notice instead.",
              "review_date": "2025-12-28",
              "score_overall": 8.9,
              "score_cleanliness": 8.8,
              "score_comfort": 9.0,
              "score_facilities": 8.7,
              "score_location": 9.0,
              "score_staff": 8.9,
              "score_value_for_money": 9.0
            },
            {
              "review_id": "33077",
              "review_text": "Ask similar leader interesting. Example painting while.\nProject down local about. Work third prove goal financial.",
              "review_date": "2025-12-28",
              "score_overall": 8.8,
              "score_cleanliness": 8.9,
              "score_comfort": 8.3,
              "score_facilities": 9.3,
              "score_location": 9.7,
              "score_staff": 9.1,
              "score_value_for_money": 7.0
            }
          ],
          "cleanliness": 9.1
        },
        {
          "hotel_id": "21",
          "name": "The Bosphorus Inn",
          "city": "Istanbul",
          "country": "Turkey",
          "star_rating": 5,
          "similarity_score": 0.5842448472976685,
          "reviews": [
            {
              "review_id": "21220",
              "review_text": "Say note and floor despite result.\nFish current join girl. See choose near shake.",
              "review_date": "2025-12-31",
              "score_overall": 8.8,
              "score_cleanliness": 8.2,
              "score_comfort": 9.2,
              "score_facilities": 9.1,
              "score_location": 9.6,
              "score_staff": 8.8,
              "score_value_for_money": 7.3
            },
            {
              "review_id": "4426",
              "review_text": "National deep their company six picture its. Action call pressure trouble avoid week.",
              "review_date": "2025-12-27",
              "score_overall": 8.7,
              "score_cleanliness": 8.3,
              "score_comfort": 8.3,
              "score_facilities": 9.2,
              "score_location": 9.4,
              "score_staff": 9.0,
              "score_value_for_money": 7.7
            },
            {
              "review_id": "16729",
              "review_text": "Rest lead suddenly president since. Paper argue feel return across. Put affect town risk theory.",
              "review_date": "2025-12-27",
              "score_overall": 8.9,
              "score_cleanliness": 8.9,
              "score_comfort": 9.7,
              "score_facilities": 8.2,
              "score_location": 9.0,
              "score_staff": 8.5,
              "score_value_for_money": 8.7
            }
          ],
          "cleanliness": 9.1
        },
        {
          "hotel_id": "10",
          "name": "The Maple Grove",
          "city": "Toronto",
          "country": "Canada",
          "star_rating": 5,
          "similarity_score": 0.5799972414970398,
          "reviews": [
            {
              "review_id": "38969",
              "review_text": "Fund product study week member south. Me result choose fine policy reason. Ok stuff they lot why eat.",
              "review_date": "2025-12-31",
              "score_overall": 9.0,
              "score_cleanliness": 9.1,
              "score_comfort": 8.8,
              "score_facilities": 8.9,
              "score_location": 8.7,
              "score_staff": 9.4,
              "score_value_for_money": 8.9
            },
            {
              "review_id": "31693",
              "review_text": "Under strong shoulder fine despite more chair food.\nDefense music toward cost say seek. How writer rest service mind option most dream.",
              "review_date": "2025-12-29",
              "score_overall": 9.0,
              "score_cleanliness": 8.8,
              "score_comfort": 8.5,
              "score_facilities": 8.4,
              "score_location": 9.2,
              "score_staff": 9.7,
              "score_value_for_money": 9.4
            },
            {
              "review_id": "6981",
              "review_text": "Wait production morning garden push especially. Recent keep traditional everything. Detail six charge final morning.",
              "review_date": "2025-12-28",
              "score_overall": 9.2,
              "score_cleanliness": 9.4,
              "score_comfort": 9.3,
              "score_facilities": 9.6,
              "score_location": 8.5,
              "score_staff": 9.1,
              "score_value_for_money": 9.1
            }
          ],
          "cleanliness": 9.4
        }
      ],
      "reviews": []
    }
  }}

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