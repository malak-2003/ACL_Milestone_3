# # # # # import streamlit as st
# # # # # from example_retrieval_result import example_retrieval_result

# # # # # # Import ONLY from llm.py (no retrieval imports at all)
# # # # # from llm import generate_answers_with_all_models

# # # # # st.set_page_config(page_title="Hotel QA", page_icon="🤖", layout="centered")

# # # # # st.title("🤖 Hotel QA")

# # # # # # -----------------------------
# # # # # # Input question (UI only)
# # # # # # -----------------------------
# # # # # default_q = example_retrieval_result.get("query", "Show me The Royal Compass")
# # # # # user_question = st.text_input("Enter your question", value=default_q)

# # # # # # Optional: update dummy query with what the user typed
# # # # # # (still uses same dummy nodes/reviews)
# # # # # if user_question.strip():
# # # # #     example_retrieval_result["query"] = user_question.strip()

# # # # # run_btn = st.button("Run LLMs", type="primary", use_container_width=True)

# # # # # # -----------------------------
# # # # # # Run LLMs once and store results
# # # # # # -----------------------------
# # # # # if run_btn:
# # # # #     with st.spinner("Querying HuggingFace models..."):
# # # # #         result = generate_answers_with_all_models(example_retrieval_result)

# # # # #     st.session_state["llm_result"] = result
# # # # #     st.success("Done! Choose a model below to view its answer.")

# # # # # # -----------------------------
# # # # # # Display selection + output
# # # # # # -----------------------------
# # # # # if "llm_result" in st.session_state:
# # # # #     result = st.session_state["llm_result"]

# # # # #     st.subheader("Your Question")
# # # # #     st.write(result.get("query", ""))

# # # # #     results_dict = result.get("results", {})
# # # # #     available_models = list(results_dict.keys())

# # # # #     if not available_models:
# # # # #         st.warning("No model outputs found.")
# # # # #     else:
# # # # #         selected_model = st.selectbox(
# # # # #             "Choose a model",
# # # # #             options=available_models,
# # # # #             index=available_models.index("Qwen-2.5-7B") if "Qwen-2.5-7B" in available_models else 0
# # # # #         )

# # # # #         model_out = results_dict.get(selected_model, {})
# # # # #         status = model_out.get("status", "unknown")

# # # # #         st.subheader(f"Answer from: {selected_model}")

# # # # #         if status == "success":
# # # # #             st.write(model_out.get("answer", ""))

# # # # #             # Optional metadata
# # # # #             col1, col2 = st.columns(2)
# # # # #             col1.caption(f"Response time: {model_out.get('response_time', '?')}s")
# # # # #             qm = model_out.get("quality_metrics", {})
# # # # #             col2.caption(f"Overall quality: {qm.get('overall_quality', 'N/A')}")

# # # # #         else:
# # # # #             st.error(model_out.get("error", "Unknown error"))

# # # # import streamlit as st
# # # # from example_retrieval_result import example_retrieval_result

# # # # # Import ONLY from llm.py (no retrieval imports at all)
# # # # from llm import generate_answers_with_all_models

# # # # st.set_page_config(page_title="Hotel QA", page_icon="🤖", layout="wide")

# # # # st.title("🤖 Hotel QA System")
# # # # st.markdown("Graph-RAG powered hotel question answering with multiple LLMs")

# # # # # -----------------------------
# # # # # Input question (UI only)
# # # # # -----------------------------
# # # # default_q = example_retrieval_result.get("query", "Show me The Royal Compass")
# # # # user_question = st.text_input("Enter your question", value=default_q)

# # # # # Optional: update dummy query with what the user typed
# # # # # (still uses same dummy nodes/reviews)
# # # # if user_question.strip():
# # # #     example_retrieval_result["query"] = user_question.strip()

# # # # run_btn = st.button("🚀 Run LLMs", type="primary", use_container_width=True)

# # # # # -----------------------------
# # # # # Run LLMs once and store results
# # # # # -----------------------------
# # # # if run_btn:
# # # #     with st.spinner("Querying HuggingFace models..."):
# # # #         result = generate_answers_with_all_models(example_retrieval_result)

# # # #     st.session_state["llm_result"] = result
# # # #     st.success("✅ Done! Choose a model below to view its answer.")

# # # # # -----------------------------
# # # # # Display selection + output
# # # # # -----------------------------
# # # # if "llm_result" in st.session_state:
# # # #     result = st.session_state["llm_result"]

# # # #     st.divider()
    
# # # #     # Question display
# # # #     st.subheader("📝 Your Question")
# # # #     st.info(result.get("query", ""))

# # # #     results_dict = result.get("results", {})
# # # #     available_models = list(results_dict.keys())

# # # #     if not available_models:
# # # #         st.warning("No model outputs found.")
# # # #     else:
# # # #         # Model selection
# # # #         col1, col2 = st.columns([3, 1])
# # # #         with col1:
# # # #             selected_model = st.selectbox(
# # # #                 "Choose a model",
# # # #                 options=available_models,
# # # #                 index=available_models.index("Qwen-2.5-7B") if "Qwen-2.5-7B" in available_models else 0
# # # #             )
        
# # # #         with col2:
# # # #             # Toggle for KG context
# # # #             show_kg = st.checkbox("Show KG Context", value=False)

# # # #         model_out = results_dict.get(selected_model, {})
# # # #         status = model_out.get("status", "unknown")

# # # #         st.divider()
        
# # # #         # Display KG Context if toggled
# # # #         if show_kg:
# # # #             st.subheader("🔍 Knowledge Graph Retrieved Context")
# # # #             st.caption("Raw information retrieved from the knowledge graph before LLM processing")
            
# # # #             context = result.get("context", "")
# # # #             if context:
# # # #                 with st.expander("View Full KG Context", expanded=True):
# # # #                     st.text(context)
                
# # # #                 # Show retrieval stats
# # # #                 stats = result.get("retrieval_stats", {})
# # # #                 col1, col2, col3 = st.columns(3)
# # # #                 col1.metric("Total Hotels", stats.get("total_nodes", 0))
# # # #                 col2.metric("Total Reviews", stats.get("total_reviews", 0))
                
# # # #                 # Source breakdown
# # # #                 source_breakdown = stats.get("source_breakdown", {})
# # # #                 if source_breakdown:
# # # #                     with col3:
# # # #                         st.caption("**Source Breakdown:**")
# # # #                         for source, count in source_breakdown.items():
# # # #                             st.caption(f"• {source}: {count}")
# # # #             else:
# # # #                 st.warning("No KG context available")
            
# # # #             st.divider()

# # # #         # Display LLM Answer
# # # #         st.subheader(f"💬 Answer from: {selected_model}")

# # # #         if status == "success":
# # # #             # Main answer in a nice container
# # # #             st.markdown(f"**{model_out.get('answer', '')}**")

# # # #             # Metadata
# # # #             st.divider()
# # # #             col1, col2, col3, col4 = st.columns(4)
            
# # # #             with col1:
# # # #                 st.metric("Response Time", f"{model_out.get('response_time', '?')}s")
            
# # # #             qm = model_out.get("quality_metrics", {})
# # # #             with col2:
# # # #                 quality_score = qm.get('overall_quality', 0)
# # # #                 st.metric("Overall Quality", f"{quality_score:.1%}")
            
# # # #             with col3:
# # # #                 st.metric("Word Count", qm.get('length_words', 'N/A'))
            
# # # #             with col4:
# # # #                 st.metric("Coverage", f"{qm.get('coverage', 0):.1%}")

# # # #             # Additional quality metrics in expander
# # # #             with st.expander("📊 Detailed Quality Metrics"):
# # # #                 col1, col2, col3 = st.columns(3)
                
# # # #                 with col1:
# # # #                     st.caption("**Accuracy Metrics**")
# # # #                     st.write(f"Accuracy Score: {qm.get('accuracy_score', 0):.1%}")
# # # #                     st.write(f"Grounding Score: {qm.get('grounding_score', 0):.1%}")
# # # #                     st.write(f"✓ Mentions Ratings" if qm.get('mentions_ratings') else "✗ Mentions Ratings")
# # # #                     st.write(f"✓ Mentions Location" if qm.get('mentions_location') else "✗ Mentions Location")
                
# # # #                 with col2:
# # # #                     st.caption("**Relevance Metrics**")
# # # #                     st.write(f"Relevance Score: {qm.get('relevance_score', 0):.1%}")
# # # #                     st.write(f"Query Relevance: {qm.get('query_relevance', 0):.1%}")
# # # #                     st.write(f"✓ Provides Comparison" if qm.get('provides_comparison') else "✗ Provides Comparison")
# # # #                     st.write(f"✓ Actionable" if qm.get('actionable') else "✗ Actionable")
                
# # # #                 with col3:
# # # #                     st.caption("**Naturalness Metrics**")
# # # #                     st.write(f"Naturalness Score: {qm.get('naturalness_score', 0):.1%}")
# # # #                     st.write(f"Info Density: {qm.get('info_density', 0):.1f} words/sentence")
# # # #                     st.write(f"✓ Conversational" if qm.get('conversational_tone') else "✗ Conversational")
# # # #                     st.write(f"Completeness: {qm.get('completeness', 'N/A').replace('_', ' ').title()}")

# # # #         else:
# # # #             st.error(f"❌ Error: {model_out.get('error', 'Unknown error')}")

# # # #         # Compare all models button
# # # #         st.divider()
# # # #         if st.button("📊 Compare All Models", use_container_width=True):
# # # #             st.subheader("Model Comparison")
            
# # # #             comparison_data = []
# # # #             for model_name in available_models:
# # # #                 m_out = results_dict.get(model_name, {})
# # # #                 if m_out.get("status") == "success":
# # # #                     m_qm = m_out.get("quality_metrics", {})
# # # #                     comparison_data.append({
# # # #                         "Model": model_name,
# # # #                         "Quality": f"{m_qm.get('overall_quality', 0):.1%}",
# # # #                         "Response Time": f"{m_out.get('response_time', 0):.2f}s",
# # # #                         "Word Count": m_qm.get('length_words', 0),
# # # #                         "Accuracy": f"{m_qm.get('accuracy_score', 0):.1%}",
# # # #                         "Relevance": f"{m_qm.get('relevance_score', 0):.1%}"
# # # #                     })
            
# # # #             if comparison_data:
# # # #                 st.table(comparison_data)
# # # #             else:
# # # #                 st.warning("No successful 

# # # import streamlit as st
# # # from example_retrieval_result import example_retrieval_result
# # # from llm import generate_answers_with_all_models, get_ui_response

# # # st.set_page_config(page_title="Hotel Assistant", page_icon="🏨", layout="wide")

# # # # Custom CSS for chat-like interface
# # # st.markdown("""
# # # <style>
# # #     .user-message {
# # #         background-color: #e3f2fd;
# # #         padding: 15px;
# # #         border-radius: 15px;
# # #         margin: 10px 0;
# # #         border-left: 4px solid #2196F3;
# # #     }
# # #     .assistant-message {
# # #         background-color: #f5f5f5;
# # #         padding: 15px;
# # #         border-radius: 15px;
# # #         margin: 10px 0;
# # #         border-left: 4px solid #4CAF50;
# # #     }
# # #     .kg-context {
# # #         background-color: #fff3e0;
# # #         padding: 15px;
# # #         border-radius: 10px;
# # #         margin: 10px 0;
# # #         border-left: 4px solid #ff9800;
# # #         font-family: monospace;
# # #         font-size: 12px;
# # #     }
# # #     .model-badge {
# # #         display: inline-block;
# # #         padding: 5px 10px;
# # #         border-radius: 5px;
# # #         margin: 5px;
# # #         font-size: 12px;
# # #         font-weight: bold;
# # #     }
# # #     .metric-card {
# # #         background-color: white;
# # #         padding: 10px;
# # #         border-radius: 8px;
# # #         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
# # #         text-align: center;
# # #     }
# # # </style>
# # # """, unsafe_allow_html=True)

# # # # Initialize session state for chat history
# # # if "chat_history" not in st.session_state:
# # #     st.session_state.chat_history = []

# # # # Title and description
# # # st.title("🏨 Hotel Assistant Chatbot")
# # # st.markdown("*Ask me anything about hotels! I'll search the knowledge graph and provide answers using multiple AI models.*")

# # # # Sidebar for settings
# # # with st.sidebar:
# # #     st.header("⚙️ Settings")
    
# # #     # Model selection
# # #     st.subheader("Select Models to Use")
# # #     use_llama_3b = st.checkbox("Llama-3.2-3B", value=True)
# # #     use_llama_1b = st.checkbox("Llama-3.2-1B", value=True)
# # #     use_qwen = st.checkbox("Qwen-2.5-7B", value=True)
# # #     use_gemma = st.checkbox("Gemma-2B", value=False)
    
# # #     st.divider()
    
# # #     # Display options
# # #     st.subheader("Display Options")
# # #     show_kg_default = st.checkbox("Show KG Context by default", value=False)
# # #     show_metrics = st.checkbox("Show Quality Metrics", value=True)
    
# # #     st.divider()
    
# # #     # Chat controls
# # #     if st.button("🗑️ Clear Chat History", use_container_width=True):
# # #         st.session_state.chat_history = []
# # #         if "llm_result" in st.session_state:
# # #             del st.session_state["llm_result"]
# # #         st.rerun()

# # # # Main chat interface
# # # st.divider()

# # # # Display chat history
# # # for message in st.session_state.chat_history:
# # #     if message["role"] == "user":
# # #         st.markdown(f'<div class="user-message">👤 <b>You:</b><br>{message["content"]}</div>', 
# # #                    unsafe_allow_html=True)
# # #     elif message["role"] == "assistant":
# # #         st.markdown(f'<div class="assistant-message">🤖 <b>Assistant ({message.get("model", "AI")}):</b><br>{message["content"]}</div>', 
# # #                    unsafe_allow_html=True)
# # #     elif message["role"] == "kg_context":
# # #         with st.expander("🔍 Knowledge Graph Context", expanded=message.get("expanded", False)):
# # #             st.text(message["content"])
            
# # #             # Show retrieval stats
# # #             if "stats" in message:
# # #                 stats = message["stats"]
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1:
# # #                     st.metric("Hotels Retrieved", stats.get("total_nodes", 0))
# # #                 with col2:
# # #                     st.metric("Reviews Found", stats.get("total_reviews", 0))
# # #                 with col3:
# # #                     source_breakdown = stats.get("source_breakdown", {})
# # #                     st.caption("**Sources:**")
# # #                     for source, count in source_breakdown.items():
# # #                         st.caption(f"• {source}: {count}")

# # # # Input area
# # # with st.container():
# # #     col1, col2 = st.columns([5, 1])
    
# # #     with col1:
# # #         user_input = st.text_input(
# # #             "Type your question here...",
# # #             placeholder="e.g., Find me a 5-star hotel in Cairo",
# # #             key="user_input",
# # #             label_visibility="collapsed"
# # #         )
    
# # #     with col2:
# # #         send_button = st.button("📤 Send", use_container_width=True, type="primary")

# # # # Process user input
# # # if send_button and user_input.strip():
# # #     # Add user message to chat history
# # #     st.session_state.chat_history.append({
# # #         "role": "user",
# # #         "content": user_input.strip()
# # #     })
    
# # #     # Update example retrieval result with user query
# # #     example_retrieval_result["query"] = user_input.strip()
    
# # #     # Show processing message
# # #     with st.spinner("🔍 Searching knowledge graph and querying AI models..."):
# # #         result = generate_answers_with_all_models(example_retrieval_result)
# # #         st.session_state["llm_result"] = result
    
# # #     # Add KG context to chat history
# # #     st.session_state.chat_history.append({
# # #         "role": "kg_context",
# # #         "content": result.get("context", ""),
# # #         "stats": result.get("retrieval_stats", {}),
# # #         "expanded": show_kg_default
# # #     })
    
# # #     # Add model responses to chat history
# # #     results_dict = result.get("results", {})
# # #     for model_name, output in results_dict.items():
# # #         # Skip models not selected
# # #         if model_name == "Llama-3.2-3B" and not use_llama_3b:
# # #             continue
# # #         if model_name == "Llama-3.2-1B" and not use_llama_1b:
# # #             continue
# # #         if model_name == "Qwen-2.5-7B" and not use_qwen:
# # #             continue
# # #         if model_name == "Gemma-2B" and not use_gemma:
# # #             continue
        
# # #         if output.get("status") == "success":
# # #             st.session_state.chat_history.append({
# # #                 "role": "assistant",
# # #                 "model": model_name,
# # #                 "content": output.get("answer", ""),
# # #                 "metrics": output.get("quality_metrics", {}) if show_metrics else None,
# # #                 "response_time": output.get("response_time", 0)
# # #             })
    
# # #     st.rerun()

# # # # Display model comparison if results exist
# # # if "llm_result" in st.session_state and show_metrics:
# # #     result = st.session_state["llm_result"]
# # #     results_dict = result.get("results", {})
    
# # #     # Filter successful results
# # #     successful_models = {
# # #         name: output for name, output in results_dict.items()
# # #         if output.get("status") == "success"
# # #     }
    
# # #     if len(successful_models) > 1:
# # #         st.divider()
        
# # #         with st.expander("📊 Model Comparison & Analytics", expanded=False):
# # #             st.subheader("Performance Comparison")
            
# # #             # Create comparison table
# # #             comparison_data = []
# # #             for model_name, output in successful_models.items():
# # #                 metrics = output.get("quality_metrics", {})
# # #                 comparison_data.append({
# # #                     "Model": model_name,
# # #                     "Quality Score": f"{metrics.get('overall_quality', 0):.1%}",
# # #                     "Response Time": f"{output.get('response_time', 0):.2f}s",
# # #                     "Words": metrics.get('length_words', 0),
# # #                     "Accuracy": f"{metrics.get('accuracy_score', 0):.1%}",
# # #                     "Naturalness": f"{metrics.get('naturalness_score', 0):.1%}"
# # #                 })
            
# # #             st.table(comparison_data)
            
# # #             # Best model recommendations
# # #             st.subheader("🏆 Recommendations")
            
# # #             # Find best models
# # #             qualities = {name: output.get("quality_metrics", {}).get("overall_quality", 0) 
# # #                         for name, output in successful_models.items()}
# # #             times = {name: output.get("response_time", 999) 
# # #                     for name, output in successful_models.items()}
# # #             accuracies = {name: output.get("quality_metrics", {}).get("accuracy_score", 0) 
# # #                          for name, output in successful_models.items()}
            
# # #             best_quality = max(qualities, key=qualities.get)
# # #             fastest = min(times, key=times.get)
# # #             most_accurate = max(accuracies, key=accuracies.get)
            
# # #             col1, col2, col3 = st.columns(3)
            
# # #             with col1:
# # #                 st.markdown(f"""
# # #                 <div class="metric-card">
# # #                     <h4>🥇 Best Quality</h4>
# # #                     <p><b>{best_quality}</b></p>
# # #                     <p style="font-size: 12px; color: #666;">{qualities[best_quality]:.1%}</p>
# # #                 </div>
# # #                 """, unsafe_allow_html=True)
            
# # #             with col2:
# # #                 st.markdown(f"""
# # #                 <div class="metric-card">
# # #                     <h4>⚡ Fastest</h4>
# # #                     <p><b>{fastest}</b></p>
# # #                     <p style="font-size: 12px; color: #666;">{times[fastest]:.2f}s</p>
# # #                 </div>
# # #                 """, unsafe_allow_html=True)
            
# # #             with col3:
# # #                 st.markdown(f"""
# # #                 <div class="metric-card">
# # #                     <h4>🎯 Most Accurate</h4>
# # #                     <p><b>{most_accurate}</b></p>
# # #                     <p style="font-size: 12px; color: #666;">{accuracies[most_accurate]:.1%}</p>
# # #                 </div>
# # #                 """, unsafe_allow_html=True)
            
# # #             # Detailed metrics for each model
# # #             st.subheader("Detailed Metrics by Model")
            
# # #             for model_name, output in successful_models.items():
# # #                 with st.expander(f"📈 {model_name} Detailed Metrics"):
# # #                     metrics = output.get("quality_metrics", {})
                    
# # #                     col1, col2, col3, col4 = st.columns(4)
                    
# # #                     with col1:
# # #                         st.metric("Overall Quality", f"{metrics.get('overall_quality', 0):.1%}")
# # #                         st.metric("Response Time", f"{output.get('response_time', 0):.2f}s")
                    
# # #                     with col2:
# # #                         st.metric("Accuracy", f"{metrics.get('accuracy_score', 0):.1%}")
# # #                         st.metric("Grounding", f"{metrics.get('grounding_score', 0):.1%}")
                    
# # #                     with col3:
# # #                         st.metric("Relevance", f"{metrics.get('relevance_score', 0):.1%}")
# # #                         st.metric("Coverage", f"{metrics.get('coverage', 0):.1%}")
                    
# # #                     with col4:
# # #                         st.metric("Naturalness", f"{metrics.get('naturalness_score', 0):.1%}")
# # #                         st.metric("Word Count", metrics.get('length_words', 0))
                    
# # #                     st.caption("**Content Features:**")
# # #                     features = []
# # #                     if metrics.get('mentions_ratings'):
# # #                         features.append("✓ Mentions Ratings")
# # #                     if metrics.get('mentions_location'):
# # #                         features.append("✓ Mentions Location")
# # #                     if metrics.get('provides_comparison'):
# # #                         features.append("✓ Provides Comparison")
# # #                     if metrics.get('actionable'):
# # #                         features.append("✓ Actionable Advice")
# # #                     if metrics.get('conversational_tone'):
# # #                         features.append("✓ Conversational Tone")
                    
# # #                     st.write(" | ".join(features) if features else "No special features detected")

# # # # Footer
# # # st.divider()
# # # st.caption("💡 Tip: Use the sidebar to customize which models to use and display options. Toggle 'Show KG Context' to see the raw data retrieved from the knowledge graph.")



# # import streamlit as st
# # from example_retrieval_result import example_retrieval_result
# # from llm import generate_answers_with_all_models, get_ui_response

# # st.set_page_config(page_title="Hotel Assistant", page_icon="🏨", layout="wide")

# # # Custom CSS for chat-like interface
# # st.markdown("""
# # <style>
# #     .user-message {
# #         background-color: #e3f2fd;
# #         padding: 15px;
# #         border-radius: 15px;
# #         margin: 10px 0;
# #         border-left: 4px solid #2196F3;
# #     }
# #     .assistant-message {
# #         background-color: #f5f5f5;
# #         padding: 15px;
# #         border-radius: 15px;
# #         margin: 10px 0;
# #         border-left: 4px solid #4CAF50;
# #     }
# #     .kg-context {
# #         background-color: #fff3e0;
# #         padding: 15px;
# #         border-radius: 10px;
# #         margin: 10px 0;
# #         border-left: 4px solid #ff9800;
# #         font-family: monospace;
# #         font-size: 12px;
# #     }
# #     .model-badge {
# #         display: inline-block;
# #         padding: 5px 10px;
# #         border-radius: 5px;
# #         margin: 5px;
# #         font-size: 12px;
# #         font-weight: bold;
# #     }
# #     .metric-card {
# #         background-color: white;
# #         padding: 10px;
# #         border-radius: 8px;
# #         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
# #         text-align: center;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # # Initialize session state for chat history
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []

# # # Title and description
# # st.title("🏨 Hotel Assistant Chatbot")
# # st.markdown("*Ask me anything about hotels! I'll search the knowledge graph and provide answers using multiple AI models.*")

# # # Sidebar for settings
# # with st.sidebar:
# #     st.header("⚙️ Settings")
    
# #     # Model selection
# #     st.subheader("Select Models to Use")
# #     use_llama_3b = st.checkbox("Llama-3.2-3B", value=True)
# #     use_llama_1b = st.checkbox("Llama-3.2-1B", value=True)
# #     use_qwen = st.checkbox("Qwen-2.5-7B", value=True)
# #     use_gemma = st.checkbox("Gemma-2B", value=False)
    
# #     st.divider()
    
# #     # Display options
# #     st.subheader("Display Options")
# #     show_kg_default = st.checkbox("Show KG Context by default", value=False)
# #     show_metrics = st.checkbox("Show Quality Metrics", value=True)
    
# #     st.divider()
    
# #     # Chat controls
# #     if st.button("🗑️ Clear Chat History", use_container_width=True):
# #         st.session_state.chat_history = []
# #         if "llm_result" in st.session_state:
# #             del st.session_state["llm_result"]
# #         st.rerun()

# # # Main chat interface
# # st.divider()

# # # Display chat history
# # for message in st.session_state.chat_history:
# #     if message["role"] == "user":
# #         st.markdown(f'<div class="user-message">👤 <b>You:</b><br>{message["content"]}</div>', 
# #                    unsafe_allow_html=True)
# #     elif message["role"] == "assistant":
# #         st.markdown(f'<div class="assistant-message">🤖 <b>Assistant ({message.get("model", "AI")}):</b><br>{message["content"]}</div>', 
# #                    unsafe_allow_html=True)
# #     elif message["role"] == "kg_context":
# #         title = message.get("title", "Knowledge Graph Retrieved Data")
# #         with st.expander(f"🔍 {title}", expanded=message.get("expanded", False)):
# #             st.json(message["content"]) if message["content"].startswith('{') or message["content"].startswith('[') else st.code(message["content"], language="json")
            
# #             # Show retrieval stats
# #             if "stats" in message:
# #                 st.divider()
# #                 st.caption("**Retrieval Statistics:**")
# #                 stats = message["stats"]
# #                 col1, col2, col3 = st.columns(3)
# #                 with col1:
# #                     st.metric("Hotels Retrieved", stats.get("total_nodes", 0))
# #                 with col2:
# #                     st.metric("Reviews Found", stats.get("total_reviews", 0))
# #                 with col3:
# #                     source_breakdown = stats.get("source_breakdown", {})
# #                     st.caption("**Sources:**")
# #                     for source, count in source_breakdown.items():
# #                         st.caption(f"• {source}: {count}")

# # # Input area
# # with st.container():
# #     col1, col2 = st.columns([5, 1])
    
# #     with col1:
# #         user_input = st.text_input(
# #             "Type your question here...",
# #             placeholder="e.g., Find me a 5-star hotel in Cairo",
# #             key="user_input",
# #             label_visibility="collapsed"
# #         )
    
# #     with col2:
# #         send_button = st.button("📤 Send", use_container_width=True, type="primary")

# # # Process user input
# # if send_button and user_input.strip():
# #     # Add user message to chat history
# #     st.session_state.chat_history.append({
# #         "role": "user",
# #         "content": user_input.strip()
# #     })
    
# #     # Update example retrieval result with user query
# #     example_retrieval_result["query"] = user_input.strip()
    
# #     # Show processing message
# #     with st.spinner("🔍 Searching knowledge graph and querying AI models..."):
# #         result = generate_answers_with_all_models(example_retrieval_result)
# #         st.session_state["llm_result"] = result
    
# #     # Add RAW KG data to chat history (before LLM processing)
# #     raw_kg_data = example_retrieval_result.get("results", {})
    
# #     # Format raw KG data for display
# #     import json
# #     raw_kg_display = json.dumps(raw_kg_data, indent=2)
    
# #     st.session_state.chat_history.append({
# #         "role": "kg_context",
# #         "content": raw_kg_display,
# #         "stats": result.get("retrieval_stats", {}),
# #         "expanded": show_kg_default,
# #         "title": "Raw Knowledge Graph Data (Before LLM Processing)"
# #     })
    
# #     # Add model responses to chat history
# #     results_dict = result.get("results", {})
# #     for model_name, output in results_dict.items():
# #         # Skip models not selected
# #         if model_name == "Llama-3.2-3B" and not use_llama_3b:
# #             continue
# #         if model_name == "Llama-3.2-1B" and not use_llama_1b:
# #             continue
# #         if model_name == "Qwen-2.5-7B" and not use_qwen:
# #             continue
# #         if model_name == "Gemma-2B" and not use_gemma:
# #             continue
        
# #         if output.get("status") == "success":
# #             st.session_state.chat_history.append({
# #                 "role": "assistant",
# #                 "model": model_name,
# #                 "content": output.get("answer", ""),
# #                 "metrics": output.get("quality_metrics", {}) if show_metrics else None,
# #                 "response_time": output.get("response_time", 0)
# #             })
    
# #     st.rerun()

# # # Display model comparison if results exist
# # if "llm_result" in st.session_state and show_metrics:
# #     result = st.session_state["llm_result"]
# #     results_dict = result.get("results", {})
    
# #     # Filter successful results
# #     successful_models = {
# #         name: output for name, output in results_dict.items()
# #         if output.get("status") == "success"
# #     }
    
# #     if len(successful_models) > 1:
# #         st.divider()
        
# #         with st.expander("📊 Model Comparison & Analytics", expanded=False):
# #             st.subheader("Performance Comparison")
            
# #             # Create comparison table
# #             comparison_data = []
# #             for model_name, output in successful_models.items():
# #                 metrics = output.get("quality_metrics", {})
# #                 comparison_data.append({
# #                     "Model": model_name,
# #                     "Quality Score": f"{metrics.get('overall_quality', 0):.1%}",
# #                     "Response Time": f"{output.get('response_time', 0):.2f}s",
# #                     "Words": metrics.get('length_words', 0),
# #                     "Accuracy": f"{metrics.get('accuracy_score', 0):.1%}",
# #                     "Naturalness": f"{metrics.get('naturalness_score', 0):.1%}"
# #                 })
            
# #             st.table(comparison_data)
            
# #             # Best model recommendations
# #             st.subheader("🏆 Recommendations")
            
# #             # Find best models
# #             qualities = {name: output.get("quality_metrics", {}).get("overall_quality", 0) 
# #                         for name, output in successful_models.items()}
# #             times = {name: output.get("response_time", 999) 
# #                     for name, output in successful_models.items()}
# #             accuracies = {name: output.get("quality_metrics", {}).get("accuracy_score", 0) 
# #                          for name, output in successful_models.items()}
            
# #             best_quality = max(qualities, key=qualities.get)
# #             fastest = min(times, key=times.get)
# #             most_accurate = max(accuracies, key=accuracies.get)
            
# #             col1, col2, col3 = st.columns(3)
            
# #             with col1:
# #                 st.markdown(f"""
# #                 <div class="metric-card">
# #                     <h4>🥇 Best Quality</h4>
# #                     <p><b>{best_quality}</b></p>
# #                     <p style="font-size: 12px; color: #666;">{qualities[best_quality]:.1%}</p>
# #                 </div>
# #                 """, unsafe_allow_html=True)
            
# #             with col2:
# #                 st.markdown(f"""
# #                 <div class="metric-card">
# #                     <h4>⚡ Fastest</h4>
# #                     <p><b>{fastest}</b></p>
# #                     <p style="font-size: 12px; color: #666;">{times[fastest]:.2f}s</p>
# #                 </div>
# #                 """, unsafe_allow_html=True)
            
# #             with col3:
# #                 st.markdown(f"""
# #                 <div class="metric-card">
# #                     <h4>🎯 Most Accurate</h4>
# #                     <p><b>{most_accurate}</b></p>
# #                     <p style="font-size: 12px; color: #666;">{accuracies[most_accurate]:.1%}</p>
# #                 </div>
# #                 """, unsafe_allow_html=True)
            
# #             # Detailed metrics for each model
# #             st.subheader("Detailed Metrics by Model")
            
# #             for model_name, output in successful_models.items():
# #                 with st.expander(f"📈 {model_name} Detailed Metrics"):
# #                     metrics = output.get("quality_metrics", {})
                    
# #                     col1, col2, col3, col4 = st.columns(4)
                    
# #                     with col1:
# #                         st.metric("Overall Quality", f"{metrics.get('overall_quality', 0):.1%}")
# #                         st.metric("Response Time", f"{output.get('response_time', 0):.2f}s")
                    
# #                     with col2:
# #                         st.metric("Accuracy", f"{metrics.get('accuracy_score', 0):.1%}")
# #                         st.metric("Grounding", f"{metrics.get('grounding_score', 0):.1%}")
                    
# #                     with col3:
# #                         st.metric("Relevance", f"{metrics.get('relevance_score', 0):.1%}")
# #                         st.metric("Coverage", f"{metrics.get('coverage', 0):.1%}")
                    
# #                     with col4:
# #                         st.metric("Naturalness", f"{metrics.get('naturalness_score', 0):.1%}")
# #                         st.metric("Word Count", metrics.get('length_words', 0))
                    
# #                     st.caption("**Content Features:**")
# #                     features = []
# #                     if metrics.get('mentions_ratings'):
# #                         features.append("✓ Mentions Ratings")
# #                     if metrics.get('mentions_location'):
# #                         features.append("✓ Mentions Location")
# #                     if metrics.get('provides_comparison'):
# #                         features.append("✓ Provides Comparison")
# #                     if metrics.get('actionable'):
# #                         features.append("✓ Actionable Advice")
# #                     if metrics.get('conversational_tone'):
# #                         features.append("✓ Conversational Tone")
                    
# #                     st.write(" | ".join(features) if features else "No special features detected")

# # # Footer
# # st.divider()
# # st.caption("💡 Tip: Use the sidebar to customize which models to use and display options. Toggle 'Show KG Context' to see the raw data retrieved from the knowledge graph.")

# import streamlit as st
# from example_retrieval_result import example_retrieval_result
# from llm import generate_answers_with_all_models, get_ui_response

# st.set_page_config(page_title="Hotel Assistant", page_icon="🏨", layout="wide")

# # Custom CSS for chat-like interface
# st.markdown("""
# <style>
#     .user-message {
#         background-color: #e3f2fd;
#         padding: 15px;
#         border-radius: 15px;
#         margin: 10px 0;
#         border-left: 4px solid #2196F3;
#     }
#     .assistant-message {
#         background-color: #f5f5f5;
#         padding: 15px;
#         border-radius: 15px;
#         margin: 10px 0;
#         border-left: 4px solid #4CAF50;
#     }
#     .kg-context {
#         background-color: #fff3e0;
#         padding: 15px;
#         border-radius: 10px;
#         margin: 10px 0;
#         border-left: 4px solid #ff9800;
#         font-family: monospace;
#         font-size: 12px;
#     }
#     .model-badge {
#         display: inline-block;
#         padding: 5px 10px;
#         border-radius: 5px;
#         margin: 5px;
#         font-size: 12px;
#         font-weight: bold;
#     }
#     .metric-card {
#         background-color: white;
#         padding: 10px;
#         border-radius: 8px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         text-align: center;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Title and description
# st.title("🏨 Hotel Assistant Chatbot")
# st.markdown("*Ask me anything about hotels! I'll search the knowledge graph and provide answers using multiple AI models.*")

# # Sidebar for settings
# with st.sidebar:
#     st.header("⚙️ Settings")
    
#     # Model selection
#     st.subheader("Select Models to Use")
#     use_llama_3b = st.checkbox("Llama-3.2-3B", value=True)
#     use_llama_1b = st.checkbox("Llama-3.2-1B", value=True)
#     use_qwen = st.checkbox("Qwen-2.5-7B", value=True)
#     use_gemma = st.checkbox("Gemma-2B", value=False)
    
#     st.divider()
    
#     # Display options
#     st.subheader("Display Options")
#     show_kg_default = st.checkbox("Show KG Context by default", value=False)
#     show_metrics = st.checkbox("Show Quality Metrics", value=True)
    
#     st.divider()
    
#     # Chat controls
#     if st.button("🗑️ Clear Chat History", use_container_width=True):
#         st.session_state.chat_history = []
#         if "llm_result" in st.session_state:
#             del st.session_state["llm_result"]
#         st.rerun()

# # Main chat interface
# st.divider()

# # Display chat history
# for message in st.session_state.chat_history:
#     if message["role"] == "user":
#         st.markdown(f'<div class="user-message">👤 <b>You:</b><br>{message["content"]}</div>', 
#                    unsafe_allow_html=True)
#     elif message["role"] == "assistant":
#         st.markdown(f'<div class="assistant-message">🤖 <b>Assistant ({message.get("model", "AI")}):</b><br>{message["content"]}</div>', 
#                    unsafe_allow_html=True)
#     elif message["role"] == "kg_context":
#         title = message.get("title", "Knowledge Graph Retrieved Data")
#         with st.expander(f"🔍 {title}", expanded=message.get("expanded", False)):
#             # Display raw KG data as JSON
#             try:
#                 import json
#                 if isinstance(message["content"], str):
#                     # If it's a string, parse it first
#                     data = json.loads(message["content"])
#                     st.json(data)
#                 else:
#                     # If it's already a dict, display directly
#                     st.json(message["content"])
#             except:
#                 # Fallback to code block if JSON parsing fails
#                 st.code(message["content"], language="json")
            
#             # Show retrieval stats
#             if "stats" in message:
#                 st.divider()
#                 st.caption("**Retrieval Statistics:**")
#                 stats = message["stats"]
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Hotels Retrieved", stats.get("total_nodes", 0))
#                 with col2:
#                     st.metric("Reviews Found", stats.get("total_reviews", 0))
#                 with col3:
#                     source_breakdown = stats.get("source_breakdown", {})
#                     st.caption("**Sources:**")
#                     for source, count in source_breakdown.items():
#                         st.caption(f"• {source}: {count}")

# # Input area
# with st.container():
#     col1, col2 = st.columns([5, 1])
    
#     with col1:
#         user_input = st.text_input(
#             "Type your question here...",
#             placeholder="e.g., Find me a 5-star hotel in Cairo",
#             key="user_input",
#             label_visibility="collapsed"
#         )
    
#     with col2:
#         send_button = st.button("📤 Send", use_container_width=True, type="primary")

# # Process user input
# if send_button and user_input.strip():
#     # Add user message to chat history
#     st.session_state.chat_history.append({
#         "role": "user",
#         "content": user_input.strip()
#     })
    
#     # Update example retrieval result with user query
#     example_retrieval_result["query"] = user_input.strip()
    
#     # Show processing message
#     with st.spinner("🔍 Searching knowledge graph and querying AI models..."):
#         result = generate_answers_with_all_models(example_retrieval_result)
#         st.session_state["llm_result"] = result
    
#     # Add RAW KG data to chat history (before LLM processing)
#     raw_kg_data = example_retrieval_result.get("results", {})
    
#     st.session_state.chat_history.append({
#         "role": "kg_context",
#         "content": raw_kg_data,  # Store as dict, not string
#         "stats": result.get("retrieval_stats", {}),
#         "expanded": show_kg_default,
#         "title": "Raw Knowledge Graph Data (Before LLM Processing)"
#     })
    
#     # Add model responses to chat history
#     results_dict = result.get("results", {})
#     for model_name, output in results_dict.items():
#         # Skip models not selected
#         if model_name == "Llama-3.2-3B" and not use_llama_3b:
#             continue
#         if model_name == "Llama-3.2-1B" and not use_llama_1b:
#             continue
#         if model_name == "Qwen-2.5-7B" and not use_qwen:
#             continue
#         if model_name == "Gemma-2B" and not use_gemma:
#             continue
        
#         if output.get("status") == "success":
#             st.session_state.chat_history.append({
#                 "role": "assistant",
#                 "model": model_name,
#                 "content": output.get("answer", ""),
#                 "metrics": output.get("quality_metrics", {}) if show_metrics else None,
#                 "response_time": output.get("response_time", 0)
#             })
    
#     st.rerun()

# # Display model comparison if results exist
# if "llm_result" in st.session_state and show_metrics:
#     result = st.session_state["llm_result"]
#     results_dict = result.get("results", {})
    
#     # Filter successful results
#     successful_models = {
#         name: output for name, output in results_dict.items()
#         if output.get("status") == "success"
#     }
    
#     if len(successful_models) > 1:
#         st.divider()
        
#         with st.expander("📊 Model Comparison & Analytics", expanded=False):
#             st.subheader("Performance Comparison")
            
#             # Create comparison table
#             comparison_data = []
#             for model_name, output in successful_models.items():
#                 metrics = output.get("quality_metrics", {})
#                 comparison_data.append({
#                     "Model": model_name,
#                     "Quality Score": f"{metrics.get('overall_quality', 0):.1%}",
#                     "Response Time": f"{output.get('response_time', 0):.2f}s",
#                     "Words": metrics.get('length_words', 0),
#                     "Accuracy": f"{metrics.get('accuracy_score', 0):.1%}",
#                     "Naturalness": f"{metrics.get('naturalness_score', 0):.1%}"
#                 })
            
#             st.table(comparison_data)
            
#             # Best model recommendations
#             st.subheader("🏆 Recommendations")
            
#             # Find best models
#             qualities = {name: output.get("quality_metrics", {}).get("overall_quality", 0) 
#                         for name, output in successful_models.items()}
#             times = {name: output.get("response_time", 999) 
#                     for name, output in successful_models.items()}
#             accuracies = {name: output.get("quality_metrics", {}).get("accuracy_score", 0) 
#                          for name, output in successful_models.items()}
            
#             best_quality = max(qualities, key=qualities.get)
#             fastest = min(times, key=times.get)
#             most_accurate = max(accuracies, key=accuracies.get)
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.markdown(f"""
#                 <div class="metric-card">
#                     <h4>🥇 Best Quality</h4>
#                     <p><b>{best_quality}</b></p>
#                     <p style="font-size: 12px; color: #666;">{qualities[best_quality]:.1%}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col2:
#                 st.markdown(f"""
#                 <div class="metric-card">
#                     <h4>⚡ Fastest</h4>
#                     <p><b>{fastest}</b></p>
#                     <p style="font-size: 12px; color: #666;">{times[fastest]:.2f}s</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             with col3:
#                 st.markdown(f"""
#                 <div class="metric-card">
#                     <h4>🎯 Most Accurate</h4>
#                     <p><b>{most_accurate}</b></p>
#                     <p style="font-size: 12px; color: #666;">{accuracies[most_accurate]:.1%}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             # Detailed metrics for each model
#             st.subheader("Detailed Metrics by Model")
            
#             for model_name, output in successful_models.items():
#                 with st.expander(f"📈 {model_name} Detailed Metrics"):
#                     metrics = output.get("quality_metrics", {})
                    
#                     col1, col2, col3, col4 = st.columns(4)
                    
#                     with col1:
#                         st.metric("Overall Quality", f"{metrics.get('overall_quality', 0):.1%}")
#                         st.metric("Response Time", f"{output.get('response_time', 0):.2f}s")
                    
#                     with col2:
#                         st.metric("Accuracy", f"{metrics.get('accuracy_score', 0):.1%}")
#                         st.metric("Grounding", f"{metrics.get('grounding_score', 0):.1%}")
                    
#                     with col3:
#                         st.metric("Relevance", f"{metrics.get('relevance_score', 0):.1%}")
#                         st.metric("Coverage", f"{metrics.get('coverage', 0):.1%}")
                    
#                     with col4:
#                         st.metric("Naturalness", f"{metrics.get('naturalness_score', 0):.1%}")
#                         st.metric("Word Count", metrics.get('length_words', 0))
                    
#                     st.caption("**Content Features:**")
#                     features = []
#                     if metrics.get('mentions_ratings'):
#                         features.append("✓ Mentions Ratings")
#                     if metrics.get('mentions_location'):
#                         features.append("✓ Mentions Location")
#                     if metrics.get('provides_comparison'):
#                         features.append("✓ Provides Comparison")
#                     if metrics.get('actionable'):
#                         features.append("✓ Actionable Advice")
#                     if metrics.get('conversational_tone'):
#                         features.append("✓ Conversational Tone")
                    
#                     st.write(" | ".join(features) if features else "No special features detected")

# # Footer
# st.divider()
# st.caption("💡 Tip: Use the sidebar to customize which models to use and display options. Toggle 'Show KG Context' to see the raw data retrieved from the knowledge graph.")




import streamlit as st
from example_retrieval_result import example_retrieval_result
from llm import generate_answers_with_all_models, get_ui_response

st.set_page_config(page_title="Hotel Assistant", page_icon="🏨", layout="wide")

# Custom CSS for chat-like interface
st.markdown("""
<style>
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .kg-context {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ff9800;
        font-family: monospace;
        font-size: 12px;
    }
    .model-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 5px;
        font-size: 12px;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("🏨 Hotel Assistant Chatbot")
st.markdown("*Ask me anything about hotels! I'll search the knowledge graph and provide answers using multiple AI models.*")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Retrieval method selection
    st.subheader("🔍 Retrieval Methods")
    st.caption("Select which retrieval method(s) to use:")
    
    use_baseline = st.checkbox("Baseline (Cypher Queries)", value=True, 
                               help="Traditional graph database queries using Cypher")
    use_minilm = st.checkbox("MiniLM Embeddings", value=True,
                            help="Semantic search using MiniLM model")
    use_mpnet = st.checkbox("MPNet Embeddings", value=True,
                           help="Semantic search using MPNet model")
    
    # Build retrieval methods list
    retrieval_methods = []
    if use_baseline:
        retrieval_methods.append("baseline")
    if use_minilm:
        retrieval_methods.append("minilm")
    if use_mpnet:
        retrieval_methods.append("mpnet")
    
    if not retrieval_methods:
        st.warning("⚠️ Select at least one retrieval method!")
    else:
        if len(retrieval_methods) == 3:
            st.success("🎯 Using Hybrid Approach (All Methods)")
        elif len(retrieval_methods) == 1:
            st.info(f"📍 Using {retrieval_methods[0].upper()} only")
        else:
            st.info(f"🔀 Using {' + '.join([m.upper() for m in retrieval_methods])}")
    
    st.divider()
    
    # Model selection
    st.subheader("🤖 LLM Models")
    st.caption("Select which AI models to query:")
    
    use_llama_3b = st.checkbox("Llama-3.2-3B", value=True)
    use_llama_1b = st.checkbox("Llama-3.2-1B", value=True)
    use_qwen = st.checkbox("Qwen-2.5-7B", value=True)
    use_gemma = st.checkbox("Gemma-2B", value=False)
    
    st.divider()
    
    # Display options
    st.subheader("📊 Display Options")
    show_kg_default = st.checkbox("Show KG Context by default", value=False)
    show_metrics = st.checkbox("Show Quality Metrics", value=True)
    
    st.divider()
    
    # Chat controls
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        if "llm_result" in st.session_state:
            del st.session_state["llm_result"]
        st.rerun()

# Main chat interface
st.divider()

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">👤 <b>You:</b><br>{message["content"]}</div>', 
                   unsafe_allow_html=True)
    elif message["role"] == "system_info":
        st.info(message["content"])
    elif message["role"] == "assistant":
        st.markdown(f'<div class="assistant-message">🤖 <b>Assistant ({message.get("model", "AI")}):</b><br>{message["content"]}</div>', 
                   unsafe_allow_html=True)
    elif message["role"] == "kg_context":
        title = message.get("title", "Knowledge Graph Retrieved Data")
        with st.expander(f"🔍 {title}", expanded=message.get("expanded", False)):
            # Display raw KG data as JSON
            try:
                import json
                if isinstance(message["content"], str):
                    # If it's a string, parse it first
                    data = json.loads(message["content"])
                    st.json(data)
                else:
                    # If it's already a dict, display directly
                    st.json(message["content"])
            except:
                # Fallback to code block if JSON parsing fails
                st.code(message["content"], language="json")
            
            # Show retrieval stats
            if "stats" in message:
                st.divider()
                st.caption("**Retrieval Statistics:**")
                stats = message["stats"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Hotels Retrieved", stats.get("total_nodes", 0))
                with col2:
                    st.metric("Reviews Found", stats.get("total_reviews", 0))
                with col3:
                    source_breakdown = stats.get("source_breakdown", {})
                    st.caption("**Sources:**")
                    for source, count in source_breakdown.items():
                        st.caption(f"• {source}: {count}")

# Input area
with st.container():
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your question here...",
            placeholder="e.g., Find me a 5-star hotel in Cairo",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True, type="primary")

# Process user input
if send_button and user_input.strip():
    # Validate retrieval method selection
    if not retrieval_methods:
        st.error("⚠️ Please select at least one retrieval method in the sidebar!")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip()
        })
        
        # Update example retrieval result with user query
        example_retrieval_result["query"] = user_input.strip()
        
        # Show processing message with selected methods
        methods_text = ", ".join([m.upper() for m in retrieval_methods])
        with st.spinner(f"🔍 Searching with {methods_text} and querying AI models..."):
            result = generate_answers_with_all_models(
                example_retrieval_result, 
                retrieval_methods=retrieval_methods
            )
            st.session_state["llm_result"] = result
        
        # Add info about retrieval methods used
        methods_used = result.get("retrieval_methods_used", [])
        if len(methods_used) == 3:
            approach_text = "🎯 Hybrid Approach (Baseline + MiniLM + MPNet)"
        elif len(methods_used) == 1:
            approach_text = f"📍 {methods_used[0].upper()} Only"
        else:
            approach_text = f"🔀 Hybrid: {' + '.join([m.upper() for m in methods_used])}"
        
        st.session_state.chat_history.append({
            "role": "system_info",
            "content": f"**Retrieval Strategy:** {approach_text}"
        })
        
        # Add RAW KG data to chat history (filtered based on selected methods)
        filtered_raw_data = result.get("filtered_raw_data", {})
        
        st.session_state.chat_history.append({
            "role": "kg_context",
            "content": filtered_raw_data,
            "stats": result.get("retrieval_stats", {}),
            "expanded": show_kg_default,
            "title": f"Raw Knowledge Graph Data ({', '.join([m.upper() for m in methods_used])})"
        })
        
        # Add model responses to chat history
        results_dict = result.get("results", {})
        for model_name, output in results_dict.items():
            # Skip models not selected
            if model_name == "Llama-3.2-3B" and not use_llama_3b:
                continue
            if model_name == "Llama-3.2-1B" and not use_llama_1b:
                continue
            if model_name == "Qwen-2.5-7B" and not use_qwen:
                continue
            if model_name == "Gemma-2B" and not use_gemma:
                continue
            
            if output.get("status") == "success":
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "model": model_name,
                    "content": output.get("answer", ""),
                    "metrics": output.get("quality_metrics", {}) if show_metrics else None,
                    "response_time": output.get("response_time", 0)
                })
        
        st.rerun()

# Display model comparison if results exist
if "llm_result" in st.session_state and show_metrics:
    result = st.session_state["llm_result"]
    results_dict = result.get("results", {})
    
    # Filter successful results
    successful_models = {
        name: output for name, output in results_dict.items()
        if output.get("status") == "success"
    }
    
    if len(successful_models) > 1:
        st.divider()
        
        with st.expander("📊 Model Comparison & Analytics", expanded=False):
            st.subheader("Performance Comparison")
            
            # Create comparison table
            comparison_data = []
            for model_name, output in successful_models.items():
                metrics = output.get("quality_metrics", {})
                comparison_data.append({
                    "Model": model_name,
                    "Quality Score": f"{metrics.get('overall_quality', 0):.1%}",
                    "Response Time": f"{output.get('response_time', 0):.2f}s",
                    "Words": metrics.get('length_words', 0),
                    "Accuracy": f"{metrics.get('accuracy_score', 0):.1%}",
                    "Naturalness": f"{metrics.get('naturalness_score', 0):.1%}"
                })
            
            st.table(comparison_data)
            
            # Best model recommendations
            st.subheader("🏆 Recommendations")
            
            # Find best models
            qualities = {name: output.get("quality_metrics", {}).get("overall_quality", 0) 
                        for name, output in successful_models.items()}
            times = {name: output.get("response_time", 999) 
                    for name, output in successful_models.items()}
            accuracies = {name: output.get("quality_metrics", {}).get("accuracy_score", 0) 
                         for name, output in successful_models.items()}
            
            best_quality = max(qualities, key=qualities.get)
            fastest = min(times, key=times.get)
            most_accurate = max(accuracies, key=accuracies.get)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🥇 Best Quality</h4>
                    <p><b>{best_quality}</b></p>
                    <p style="font-size: 12px; color: #666;">{qualities[best_quality]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚡ Fastest</h4>
                    <p><b>{fastest}</b></p>
                    <p style="font-size: 12px; color: #666;">{times[fastest]:.2f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Most Accurate</h4>
                    <p><b>{most_accurate}</b></p>
                    <p style="font-size: 12px; color: #666;">{accuracies[most_accurate]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics for each model
            st.subheader("Detailed Metrics by Model")
            
            for model_name, output in successful_models.items():
                with st.expander(f"📈 {model_name} Detailed Metrics"):
                    metrics = output.get("quality_metrics", {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Overall Quality", f"{metrics.get('overall_quality', 0):.1%}")
                        st.metric("Response Time", f"{output.get('response_time', 0):.2f}s")
                    
                    with col2:
                        st.metric("Accuracy", f"{metrics.get('accuracy_score', 0):.1%}")
                        st.metric("Grounding", f"{metrics.get('grounding_score', 0):.1%}")
                    
                    with col3:
                        st.metric("Relevance", f"{metrics.get('relevance_score', 0):.1%}")
                        st.metric("Coverage", f"{metrics.get('coverage', 0):.1%}")
                    
                    with col4:
                        st.metric("Naturalness", f"{metrics.get('naturalness_score', 0):.1%}")
                        st.metric("Word Count", metrics.get('length_words', 0))
                    
                    st.caption("**Content Features:**")
                    features = []
                    if metrics.get('mentions_ratings'):
                        features.append("✓ Mentions Ratings")
                    if metrics.get('mentions_location'):
                        features.append("✓ Mentions Location")
                    if metrics.get('provides_comparison'):
                        features.append("✓ Provides Comparison")
                    if metrics.get('actionable'):
                        features.append("✓ Actionable Advice")
                    if metrics.get('conversational_tone'):
                        features.append("✓ Conversational Tone")
                    
                    st.write(" | ".join(features) if features else "No special features detected")

# Footer
st.divider()
st.caption("💡 Tip: Use the sidebar to customize which models to use and display options. Toggle 'Show KG Context' to see the raw data retrieved from the knowledge graph.")