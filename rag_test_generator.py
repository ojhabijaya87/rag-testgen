# TfL Journey Planner Test Generator (Enhanced UI/UX with Ollama + Zephyr Support)

import streamlit as st
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import requests
import pyperclip
import json

# ---------------------------- MODEL CONFIG ---------------------------- #
MODEL_CONFIG = {
    "Select a model": {},  # Default/placeholder
    "ollama-phi (Offline via Ollama)": {},
    "ollama-llama3 (Offline via Ollama)": {},
    "ollama-mistral (Offline via Ollama)": {},
    "ollama-zephyr (Offline via Ollama)": {},
    "zephyr-7b-beta (Hugging Face Hosted)": {
        "endpoint_url": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        "temperature": 0.2,
        "max_new_tokens": 3072,
        "task": "text-generation"
    },
}

# ---------------------------- EMBEDDINGS ---------------------------- #
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------- LLM HANDLER ---------------------------- #
@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    with st.spinner("🔄 Downloading and loading model, please wait..."):
        try:
            if model_name == "Select a model":
                st.warning("⚠️ Please select a model before proceeding.")
                return lambda _: "[No model selected]"

            elif model_name.startswith("ollama-"):
                model_id = model_name.replace("ollama-", "").split(" ")[0]
                st.info(f"🧠 Using Ollama model: {model_id} (ensure it's running via `ollama run {model_id}`)")

                def ollama_generate(prompt):
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model_id, "prompt": prompt},
                        headers={"Accept": "application/json"},
                        stream=True
                    )
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                json_data = json.loads(line.decode("utf-8"))
                                full_response += json_data.get("response", "")
                            except json.JSONDecodeError:
                                continue
                    return full_response

                return ollama_generate

            elif model_name == "zephyr-7b-beta (Hugging Face Hosted)":
                st.info(f"🌐 Connecting to Hugging Face model: {model_name}")
                config = MODEL_CONFIG[model_name]
                return HuggingFaceEndpoint(
                    huggingfacehub_api_token=st.secrets.get("HF_API_KEY", ""),
                    **config
                )

            else:
                st.warning("⚠️ Unsupported model selected.")
                return lambda _: "[Unsupported model]"

        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            return lambda _: "[Error during model generation]"

# ---------------------------- STREAMLIT UI ---------------------------- #
st.set_page_config(page_title="TfL Test Generator", layout="wide")
st.title("🚇 TfL Journey Planner Test Generator")

embeddings = load_embeddings()

# Sidebar: Model and URL input
with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.selectbox("Select Model", list(MODEL_CONFIG.keys()), index=0)
    st.session_state.selected_model = model_name

    if model_name == "Select a model":
        st.warning("⚠️ Please choose a model to proceed.")
    elif model_name.startswith("ollama-"):
        model_key = model_name.replace("ollama-", "").split(" ")[0]
        st.info(f"ℹ️ Ensure Ollama is running: `ollama run {model_key}`")
    elif "Hugging Face" in model_name:
        st.info("🌍 Uses Hugging Face API. Requires internet + `HF_API_KEY`")

    st.subheader("🔗 TfL Website URL")
    url_input = st.text_area("Enter one or more TfL URLs:", value="https://tfl.gov.uk/plan-a-journey/")

    if st.button("Process URL(s)"):
        with st.spinner("🔄 Loading and processing URLs..."):
            docs = []
            for url in url_input.splitlines():
                loader = SeleniumURLLoader(urls=[url.strip()])
                st.info(f"🌐 Fetching content from: {url.strip()}")
                docs += loader.load()
            if docs:
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
                st.success(f"✅ Loaded {len(docs)} document chunks.")

# Prompt Template
TEST_PROMPT = PromptTemplate.from_template("""
As an expert QA engineer, generate comprehensive test cases in Gherkin format based on:

### APPLICATION CONTEXT:
{context}

### USER STORY:
{user_story}

### INSTRUCTIONS:
1. Cover all acceptance criteria with at least one test each
2. Write clear Gherkin syntax: Given, When, Then
3. Make the scenarios business-readable and unambiguous
""")

# Main Form
with st.form("input_form"):
    st.subheader("🧾 User Story")
    user_story = st.text_area("Paste user story + acceptance criteria", height=250)
    submitted = st.form_submit_button("Generate Test Cases")

    if submitted:
        if model_name == "Select a model":
            st.warning("⚠️ Please select a model to proceed.")
        elif user_story.strip() == "":
            st.warning("⚠️ Please enter a user story.")
        else:
            st.info("🧠 Generating test cases... Please wait")
            llm = get_llm(model_name)
            feature_name = "Journey Planning"

            context = ""
            if "vector_store" in st.session_state and st.session_state.vector_store:
                with st.spinner("🔍 Retrieving relevant content from loaded documents..."):
                    context_docs = st.session_state.vector_store.similarity_search(feature_name, k=4)
                    context = "\n\n".join([doc.page_content for doc in context_docs])
                    st.success("📄 Context successfully retrieved!")

            prompt = TEST_PROMPT.format(context=context, user_story=user_story, feature_name=feature_name)

            with st.spinner("✍️ Generating test cases using selected model..."):
                response = llm(prompt) if callable(llm) else llm.invoke(prompt)

            st.session_state.generated_tests = response
            st.success("✅ Test cases generated successfully!")

# Output Section
if "generated_tests" in st.session_state:
    st.subheader("🧪 Generated Test Cases")
    st.code(st.session_state.generated_tests, language="gherkin")
    if st.button("📋 Copy to Clipboard"):
        pyperclip.copy(st.session_state.generated_tests)
        st.toast("📋 Copied to clipboard!")

# Installation Guide
with st.expander("🛠 Installation Guide", expanded=False):
    st.markdown("""
### Requirements
- Python 3.9 or later
- Ollama (for local models): https://ollama.com
- Hugging Face API key (for hosted models)

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the app:
```bash
streamlit run app.py
```

### Run a model locally with Ollama:
```bash
ollama run zephyr
```

### Hugging Face API Key
Set your key in `.streamlit/secrets.toml`:
```toml
HF_API_KEY = "your-huggingface-token"
```
""")
