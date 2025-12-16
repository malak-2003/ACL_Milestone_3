import streamlit as st
from example_retrieval_result import example_retrieval_result

# Import ONLY from llm.py (no retrieval imports at all)
from llm import generate_answers_with_all_models

st.set_page_config(page_title="Hotel QA", page_icon="🤖", layout="centered")

st.title("🤖 Hotel QA")

# -----------------------------
# Input question (UI only)
# -----------------------------
default_q = example_retrieval_result.get("query", "Show me The Royal Compass")
user_question = st.text_input("Enter your question", value=default_q)

# Optional: update dummy query with what the user typed
# (still uses same dummy nodes/reviews)
if user_question.strip():
    example_retrieval_result["query"] = user_question.strip()

run_btn = st.button("Run LLMs", type="primary", use_container_width=True)

# -----------------------------
# Run LLMs once and store results
# -----------------------------
if run_btn:
    with st.spinner("Querying HuggingFace models..."):
        result = generate_answers_with_all_models(example_retrieval_result)

    st.session_state["llm_result"] = result
    st.success("Done! Choose a model below to view its answer.")

# -----------------------------
# Display selection + output
# -----------------------------
if "llm_result" in st.session_state:
    result = st.session_state["llm_result"]

    st.subheader("Your Question")
    st.write(result.get("query", ""))

    results_dict = result.get("results", {})
    available_models = list(results_dict.keys())

    if not available_models:
        st.warning("No model outputs found.")
    else:
        selected_model = st.selectbox(
            "Choose a model",
            options=available_models,
            index=available_models.index("Qwen-2.5-7B") if "Qwen-2.5-7B" in available_models else 0
        )

        model_out = results_dict.get(selected_model, {})
        status = model_out.get("status", "unknown")

        st.subheader(f"Answer from: {selected_model}")

        if status == "success":
            st.write(model_out.get("answer", ""))

            # Optional metadata
            col1, col2 = st.columns(2)
            col1.caption(f"Response time: {model_out.get('response_time', '?')}s")
            qm = model_out.get("quality_metrics", {})
            col2.caption(f"Overall quality: {qm.get('overall_quality', 'N/A')}")

        else:
            st.error(model_out.get("error", "Unknown error"))
