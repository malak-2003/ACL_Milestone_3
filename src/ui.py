import streamlit as st
import traceback
from datetime import datetime

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Hotel GraphRAG System", layout="wide")

st.title("🏨 Hotel GraphRAG Assistant")
st.caption("Milestone 3 – Part 4 UI (GraphRAG Demo)")

# (a) Use case reflected in interface
st.info(
    "This UI demonstrates a Graph-RAG travel assistant for the Hotel Knowledge Graph.\n"
    "You can ask about hotels, reviews, traveller demographics, and visa rules."
)

# -----------------------------
# Sidebar (controls)
# -----------------------------
st.sidebar.header("System Settings")

retrieval_method = st.sidebar.selectbox(
    "Retrieval Method",
    ["Baseline (Cypher)", "Embeddings", "Hybrid"],
    index=0
)

model_choice = st.sidebar.selectbox(
    "LLM Model",
    ["GPT-4 (dummy)", "LLaMA (dummy)", "Mistral (dummy)"],
    index=0
)

show_cypher = st.sidebar.checkbox("Show Cypher Query", value=True)
show_context = st.sidebar.checkbox("Show KG Context", value=True)

st.sidebar.divider()
st.sidebar.write("**Milestone check**: UI supports question input, KG context view, and final answer view.")

# -----------------------------
# Preset questions (d)
# -----------------------------
st.subheader("Ask a question")
preset_questions = [
    "Which hotels exceed expectations for solo female travellers aged 25-34?",
    "Show me the top 3 hotels with the highest average score_overall for business travellers.",
    "Do travellers from France need a visa to visit Egypt?",
    "Return hotels with the highest location score for women.",
]

preset_choice = st.selectbox(
    "Or choose a preset question:",
    ["(none)"] + preset_questions
)

# Main text input: user can type OR use preset
default_text = "" if preset_choice == "(none)" else preset_choice
user_question = st.text_input(
    "Enter your question:",
    value=default_text,
    placeholder="Type your own question here..."
)

run_button = st.button("Run GraphRAG", type="primary")

# -----------------------------
# Session state for history (f)
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts {time, q, context, cypher, answer}

# -----------------------------
# Dummy backend functions
# (Replace these later with teammates’ real implementation)
# -----------------------------
def backend_retrieve_context(question: str, method: str):
    """
    (e) This function represents the integration point with your RAG backend.
    Later you will replace its internals with your teammates' real retrieval logic.
    """
    # Dummy “KG-retrieved context”
    context = {
        "intent": "exceeds_expectations",
        "retrieval_method": method,
        "kg_results": [
            {"hotel": "Hotel Alpha", "age_group": "25-34", "avg_review_sum": 24.5, "base_sum": 22.0},
            {"hotel": "Hotel Beta",  "age_group": "25-34", "avg_review_sum": 26.0, "base_sum": 23.5},
        ]
    }
    return context

def backend_build_cypher(question: str):
    """
    Dummy “Cypher used” for transparency.
    Later: return the actual cypher your backend executed.
    """
    return """
MATCH (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
WHERE toLower(t.gender)='female' AND toLower(t.type)='solo'
WITH h, AVG(r.score_cleanliness + r.score_comfort + r.score_facilities) AS avg_review_sum,
     (h.cleanliness_base + h.comfort_base + h.facilities_base) AS base_sum
WHERE avg_review_sum >= base_sum
RETURN h.name AS hotel, avg_review_sum, base_sum
ORDER BY (avg_review_sum - base_sum) DESC
LIMIT 5;
""".strip()

def backend_llm_answer(question: str, context: dict, model: str):
    """
    Dummy LLM answer generator.
    Later: call the real LLM with context (RAG).
    """
    hotels = [x["hotel"] for x in context.get("kg_results", [])]
    if hotels:
        return (
            f"({model}) Based on the retrieved KG context, hotels that exceed expectations "
            f"for this demographic include: {', '.join(hotels)}."
        )
    return f"({model}) Based on the retrieved KG context, no hotels exceeded expectations."

def rag_pipeline(question: str, method: str, model: str):
    """
    (e) The UI calls ONE pipeline function that returns:
    - context (KG retrieved)
    - cypher executed
    - final answer
    This is the exact function you’ll later replace with your teammates’ pipeline.
    """
    context = backend_retrieve_context(question, method)
    cypher = backend_build_cypher(question)
    answer = backend_llm_answer(question, context, model)
    return context, cypher, answer

# -----------------------------
# Run pipeline + render results
# -----------------------------
try:
    if run_button:
        if not user_question.strip():
            st.warning("Please enter a question (or select a preset).")
        else:
            # (e) Integration with backend pipeline
            context, cypher, answer = rag_pipeline(user_question, retrieval_method, model_choice)

            # Store in history so app stays functional after answering (f)
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": user_question,
                "context": context,
                "cypher": cypher,
                "answer": answer,
            })

            st.success("Done ✅ You can ask another question anytime.")

    # Display latest result (if any)
    if st.session_state.history:
        latest = st.session_state.history[0]

        col1, col2 = st.columns(2)

        # (b) View KG-retrieved context
        with col1:
            st.subheader("🔎 KG-retrieved context")
            st.caption("This is the information retrieved from Neo4j (or embeddings) and passed to the LLM.")
            if show_context:
                st.json(latest["context"])
            else:
                st.info("Context display disabled from sidebar.")

            # Show Cypher query optionally (transparency)
            if show_cypher:
                st.subheader("🧾 Executed Cypher (debug view)")
                st.code(latest["cypher"], language="cypher")

        # (c) View the final LLM answer
        with col2:
            st.subheader("🤖 Final LLM answer")
            st.write(latest["answer"])

            st.subheader("✅ Interaction details")
            st.write(f"**Time:** {latest['time']}")
            st.write(f"**Question:** {latest['question']}")
            st.write(f"**Retrieval:** {retrieval_method} | **Model:** {model_choice}")

        # (f) App remains usable: show history + allow reruns
        with st.expander("Previous questions (history)"):
            for i, item in enumerate(st.session_state.history[:10], start=1):
                st.markdown(f"**{i}. [{item['time']}]** {item['question']}")
                st.write(item["answer"])
                st.divider()
    else:
        st.write("👆 Ask a question above and click **Run GraphRAG** to see KG context + answer.")

except Exception:
    st.error("❌ The app crashed. Here is the error:")
    st.code(traceback.format_exc())
