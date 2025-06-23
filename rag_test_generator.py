from langchain_core.prompts import PromptTemplate
import streamlit as st
import asyncio
import requests
import pyperclip
import json
import datetime
import time
import re
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODEL_CONFIG = {
    "Select a model": {},
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

MODEL_USAGE_HINTS = {
    "Select a model": "ℹ️ Please select a model to enable test generation.",
    "ollama-phi (Offline via Ollama)": "🖥️ Make sure Ollama is running locally and the 'phi' model is pulled (use `ollama pull phi`).",
    "ollama-llama3 (Offline via Ollama)": "🖥️ Make sure Ollama is running locally and the 'llama3' model is pulled (use `ollama pull llama3`).",
    "ollama-mistral (Offline via Ollama)": "🖥️ Make sure Ollama is running locally and the 'mistral' model is pulled (use `ollama pull mistral`).",
    "ollama-zephyr (Offline via Ollama)": "🖥️ Make sure Ollama is running locally and the 'zephyr' model is pulled (use `ollama pull zephyr`).",
    "zephyr-7b-beta (Hugging Face Hosted)": "🌐 Requires a valid Hugging Face API key in Streamlit secrets. Fastest for most users.",
}

OLLAMA_TIMEOUT = 600

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    if model_name == "Select a model":
        return None
    if model_name.startswith("ollama-"):
        model_id = model_name.replace("ollama-", "").split(" ")[0]
        st.info(f"🧠 Using Ollama model: {model_id}")
        st.warning("⚠️ Local models may be slower. Consider using Hugging Face for faster results.")
        def ollama_generate(prompt: str) -> str:
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_id, 
                        "prompt": prompt,
                        "stream": True
                    },
                    headers={"Accept": "application/json"},
                    timeout=OLLAMA_TIMEOUT
                )
                response.raise_for_status()
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_data = json.loads(line.decode("utf-8"))
                            if "response" in json_data:
                                full_response += json_data["response"]
                            if json_data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                return full_response
            except requests.exceptions.RequestException as e:
                st.error(f"🚨 Ollama API error: {str(e)}")
                st.error("Troubleshooting steps:\n1. Ensure Ollama is running (ollama serve)\n2. Download model: ollama pull {model_id}\n3. Check system resources")
                return f"Error: {str(e)}"
            except Exception as e:
                return f"Unexpected error: {str(e)}"
        return RunnableLambda(ollama_generate)
    elif model_name == "zephyr-7b-beta (Hugging Face Hosted)":
        st.info(f"🌐 Connecting to Hugging Face model: {model_name}")
        config = MODEL_CONFIG[model_name]
        return HuggingFaceEndpoint(
            huggingfacehub_api_token=st.secrets.get("HF_API_KEY", ""),
            **config
        )
    return None

POSITIVE_PROMPT = PromptTemplate.from_template("""
As an expert QA engineer, generate detailed BDD scenarios using appropriate Gherkin features based on the context and requirements.

OFFICIAL TfL DOCUMENTATION:
{tfl_context}

USER REQUIREMENTS:
{user_requirements}

CURRENT USER STORY:
{current_story}

INSTRUCTIONS:
1. Generate 3-5 detailed scenarios
2. Use proper Gherkin syntax with TfL terminology
3. Apply these features WHENEVER THEY IMPROVE CLARITY:
   - **Data Tables**: For scenarios with multiple input combinations
     Example: 
       When I enter journey details:
         | From     | To         | Time  |
         | Bank     | Canary Wharf | 08:00 |
         | Paddington | Heathrow   | 15:30 |
   - **Examples**: For scenario outlines with similar steps
     Example:
       Scenario Outline: Plan journey at different times
         Given I want to travel from "<from>" to "<to>"
         When I set departure time to "<time>"
         Then I see valid route options
         Examples:
           | from       | to         | time  |
           | Waterloo   | Wimbledon  | 08:00 |
           | Kings Cross | Stansted   | 05:30 |
   - **Background**: For setup steps common to multiple scenarios
     Example:
       Background:
         Given I'm on the TfL journey planner
         And I accept cookies
   - **Tags**: For test categorization
     Example: @smoke @journey_planning
4. Structure each scenario as:
   Feature: [Feature Name]
     Scenario: [Descriptive Scenario Name]
       Given [context]
       When [action]
       Then [outcome]
5. Prioritize clarity and conciseness
""")
NEGATIVE_PROMPT = PromptTemplate.from_template("""
As an expert QA engineer, generate detailed NEGATIVE BDD scenarios using appropriate Gherkin features.

OFFICIAL TfL DOCUMENTATION:
{tfl_context}

USER REQUIREMENTS:
{user_requirements}

CURRENT USER STORY:
{current_story}

INSTRUCTIONS:
1. Generate 3-5 negative scenarios
2. Focus on errors, invalid inputs, and exceptions
3. Apply these features WHENEVER THEY IMPROVE TEST EFFECTIVENESS:
   - **Data Tables**: For different invalid input combinations
     Example:
       When I enter invalid journey details:
         | From   | To     | Error Message              |
         | ""     | "Bank" | "Please enter from station"|
         | "Bank" | ""     | "Please enter to station"  |
   - **Examples**: For testing various error conditions
     Example:
       Scenario Outline: Invalid station combinations
         Given I enter "<from>" as origin
         And I enter "<to>" as destination
         When I plan journey
         Then I see "<error>"
         Examples:
           | from | to   | error                        |
           | XYZ  | Bank | "XYZ is not a valid station" |
           | Bank | XYZ  | "XYZ is not a valid station" |
   - **Background**: For common pre-conditions
4. Include specific error messages from context
5. Cover both input validation and system error cases
""")
EDGE_PROMPT = PromptTemplate.from_template("""
As an expert QA engineer, generate detailed EDGE CASE scenarios using appropriate Gherkin features.

OFFICIAL TfL DOCUMENTATION:
{tfl_context}

USER REQUIREMENTS:
{user_requirements}

CURRENT USER STORY:
{current_story}

INSTRUCTIONS:
1. Generate 3-5 edge case scenarios
2. Focus on boundary conditions and unusual situations
3. Apply these features WHERE THEY ADD VALUE:
   - **Data Tables**: For boundary value testing
     Example:
       When I plan journey on special dates:
         | Date        | Type               |
         | 2024-02-29 | Leap day           |
         | 2023-12-25 | Christmas day      |
   - **Examples**: For testing different boundary values
     Example:
       Scenario Outline: Journey at time boundaries
         Given I want to travel from "<from>" to "<to>"
         When I set time to "<time>"
         Then I see "<result>"
         Examples:
           | from | to | time  | result                |
           | A    | B  | 00:00 | Night Tube available  |
           | X    | Y  | 04:30 | First train scheduled |
   - **Tags**: For categorizing edge types (@temporal, @spatial)
4. Cover temporal, spatial, and capacity boundaries
5. Include real examples from TfL context
""")

async def generate_prompt_async(prompt: str, llm):
    try:
        if isinstance(llm, RunnableLambda):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, llm.invoke, prompt)
        else:
            return await llm.ainvoke(prompt)
    except Exception as e:
        return f"Generation Error: {str(e)}"

async def generate_all_tests(tfl_context, user_requirements, current_story, llm):
    if not llm:
        return ["No model selected", "No model selected", "No model selected"]
    prompts = [
        POSITIVE_PROMPT.format(
            tfl_context=tfl_context[:500], 
            user_requirements=user_requirements[:500],
            current_story=current_story
        ),
        NEGATIVE_PROMPT.format(
            tfl_context=tfl_context[:500], 
            user_requirements=user_requirements[:500],
            current_story=current_story
        ),
        EDGE_PROMPT.format(
            tfl_context=tfl_context[:500], 
            user_requirements=user_requirements[:500],
            current_story=current_story
        )
    ]
    tasks = [generate_prompt_async(prompt, llm) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results

def get_hybrid_context(vector_store, feature_name):
    tfl_context = ""
    user_requirements = ""
    if not vector_store:
        return tfl_context, user_requirements
    try:
        tfl_docs = vector_store.similarity_search(
            feature_name, 
            k=2,
            filter=lambda meta: meta.get("source_type") == "tfl"
        )
        tfl_context = "\n\n".join(doc.page_content[:300] for doc in tfl_docs)
        story_docs = vector_store.similarity_search(
            feature_name,
            k=1,
            filter=lambda meta: meta.get("source_type") == "user_story"
        )
        user_requirements = "\n\n".join(doc.page_content[:300] for doc in story_docs)
    except Exception as e:
        st.error(f"Context error: {str(e)}")
    return tfl_context, user_requirements

def anonymize_story(story: str) -> str:
    story = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+', 'User', story)
    story = re.sub(r'\S+@\S+', 'user@example.com', story)
    story = re.sub(r'\b\d{10}\b', 'XXXXXXXXXX', story)
    return story

def is_duplicate_story(vector_store, new_story, embeddings, threshold=0.9):
    try:
        new_emb = np.array(embeddings.embed_query(new_story)).reshape(1, -1)
        results = vector_store.similarity_search(
            new_story,
            k=5,
            filter=lambda meta: meta.get("source_type") == "user_story"
        )
        for doc in results:
            doc_emb = np.array(embeddings.embed_query(doc.page_content)).reshape(1, -1)
            sim = cosine_similarity(new_emb, doc_emb)[0][0]
            if sim >= threshold:
                return True
        return False
    except Exception as e:
        st.warning(f"Duplicate check error: {e}")
        return False

# --- Streamlit UI ---
st.set_page_config(page_title="TfL Requirements-Driven Test Generator", layout="wide")
st.title("🚇 TfL Journey Planner - Requirements-Driven Test Generator")

if "generating" not in st.session_state:
    st.session_state.generating = False

embeddings = load_embeddings()

with st.sidebar:
    st.header("⚙️ Configuration")
    model_names = list(MODEL_CONFIG.keys())
    default_index = model_names.index("zephyr-7b-beta (Hugging Face Hosted)") if "zephyr-7b-beta (Hugging Face Hosted)" in model_names else 0
    model_name = st.selectbox("Select Model", model_names, index=default_index, key="model_selector")
    st.caption(MODEL_USAGE_HINTS.get(model_name, ""))

    st.subheader("🔗 TfL Website URL")
    url_input = st.text_area("Enter one or more TfL URLs:",
                             value="https://tfl.gov.uk/plan-a-journey/",
                             key="url_input")
    store_stories = st.checkbox("📚 Store user stories for context", value=True,
                              help="Improves test relevance by remembering past requirements")
    if "vector_store" in st.session_state:
        try:
            stories = [doc for doc in st.session_state.vector_store.docstore._dict.values()
                      if doc.metadata.get("source_type") == "user_story"]
            st.metric("Stored Requirements", f"{len(stories)} user stories")
        except:
            pass
    process_disabled = st.session_state.generating
    if st.button("Process URL(s)", key="process_button", disabled=process_disabled):
        with st.spinner("🔄 Loading URLs..."):
            docs = []
            for url in url_input.splitlines():
                clean_url = url.strip()
                if not clean_url:
                    continue
                try:
                    loader = SeleniumURLLoader(urls=[clean_url])
                    loaded_docs = loader.load()
                    timestamp = datetime.datetime.now().isoformat()
                    for doc in loaded_docs:
                        doc.metadata.update({
                            "source_url": clean_url,
                            "ingested_at": timestamp,
                            "source_type": "tfl",
                        })
                    docs.extend(loaded_docs)
                    st.success(f"✅ Loaded: {clean_url}")
                except Exception as e:
                    st.error(f"❌ Failed to load {clean_url}: {str(e)}")
            if docs:
                if "vector_store" not in st.session_state:
                    st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
                else:
                    st.session_state.vector_store.add_documents(docs)
                st.success(f"📚 Processed {len(docs)} document chunks.")

with st.form("input_form"):
    st.subheader("🧾 User Story")
    user_story = st.text_area("Paste user story + acceptance criteria", 
                             height=200, 
                             placeholder="As a user, I want to plan journeys so that I can...",
                             key="user_story",
                             help="Be specific about journey planning requirements")
    if user_story:
        story_quality = min(100, len(user_story) // 2)
        st.progress(story_quality, text=f"Requirement detail: {story_quality}%")
    submitted = st.form_submit_button(
        "🚀 Generate BDD Test Cases",
        disabled=st.session_state.generating
    )
    if submitted:
        st.session_state.generating = True
        try:
            if model_name == "Select a model":
                st.warning("⚠️ Please select a model to proceed.")
            elif not user_story.strip():
                st.warning("⚠️ Please enter a user story.")
            else:
                start_time = time.perf_counter()
                with st.spinner("🧠 Generating requirements-driven test cases..."):
                    llm = get_llm(model_name)
                    feature_name = "Journey Planning"
                    if store_stories and len(user_story) > 50:
                        try:
                            story_doc = Document(
                                page_content=anonymize_story(user_story),
                                metadata={
                                    "source_type": "user_story",
                                    "feature_name": feature_name,
                                    "ingested_at": datetime.datetime.now().isoformat(),
                                }
                            )
                            if "vector_store" not in st.session_state:
                                st.session_state.vector_store = FAISS.from_documents([story_doc], embeddings)
                                st.toast("📝 Stored user story for future context", icon="✅")
                            else:
                                if not is_duplicate_story(st.session_state.vector_store, user_story, embeddings):
                                    st.session_state.vector_store.add_documents([story_doc])
                                    st.toast("📝 Stored user story for future context", icon="✅")
                                else:
                                    st.toast("⚠️ Duplicate user story not added.", icon="⚠️")
                        except Exception as e:
                            st.error(f"Failed to store user story: {str(e)}")
                    tfl_context, user_requirements = get_hybrid_context(
                        st.session_state.get("vector_store"), feature_name
                    )
                    results = asyncio.run(
                        generate_all_tests(tfl_context, user_requirements, user_story, llm)
                    )
                    positive, negative, edge = results
                    st.subheader("✅ Positive Scenarios")
                    st.code(positive, language="gherkin")
                    st.subheader("❌ Negative Scenarios")
                    st.code(negative, language="gherkin")
                    st.subheader("🟧 Edge Case Scenarios")
                    st.code(edge, language="gherkin")
                    elapsed = time.perf_counter() - start_time
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    st.info(f"⏱️ Completed in {minutes} minutes {seconds} seconds")
                    
        finally:
            st.session_state.generating = False
