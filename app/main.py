import streamlit as st

from app.chunking import chunk_text
# from app.chunking import chunk_text
# from app.embedding import embed_and_store
# from app.retrieval import load_retriever
# from app.test_case_generator import generate_test_cases

st.set_page_config(page_title="RAG Test Case Generator", layout="wide")

st.title("🧪 RAG-based Test Case Generator")
st.markdown("Paste your **requirement text, user stories, epics, or wireframes summary** below:")

# ✅ Step 1: User provides input
user_input = st.text_area("✍️ Paste Requirements Here", height=300, placeholder="e.g. As a user, I want to reset my password...")

# ✅ Step 2: Process on submit
if st.button("Generate Test Cases"):
    if user_input.strip() == "":
        st.warning("Please paste some requirement text to proceed.")
    else:
        with st.spinner("🔍 Chunking and embedding..."):
            chunks = chunk_text(user_input)
        #     retriever = embed_and_store(chunks)

        # with st.spinner("🧠 Generating test cases using RAG..."):
        #     test_cases = generate_test_cases(user_input, retriever)

        # st.success("✅ Test cases generated!")
        # st.subheader("📋 Generated Test Cases")
        # for i, tc in enumerate(test_cases, 1):
        #     st.markdown(f"**{i}.** {tc}")
