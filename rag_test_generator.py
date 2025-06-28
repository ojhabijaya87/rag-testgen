from langchain_core.prompts import PromptTemplate
import streamlit as st
import asyncio
import datetime
import time
import re
import hashlib
import os
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import numpy as np
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import additional model providers
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    # Try to import from new package
    from langchain_ollama import Ollama
except ImportError:
    # Fallback to old package
    from langchain_community.llms import Ollama

# --- UI Configuration ---
st.set_page_config(page_title="Test Case Generator", layout="wide")
st.title("🧪 AI Test Case Generator")

MODEL_CONFIG = {
    "Select a model": {},
    "ollama-phi (Offline)": {
        "provider": "ollama",
        "model_name": "phi3:14b-medium-4k-instruct",
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "ollama-llama3 (Offline)": {
        "provider": "ollama",
        "model_name": "llama3:70b-instruct",
        "temperature": 0.3,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "ollama-mistral (Offline)": {
        "provider": "ollama",
        "model_name": "mistral:7b-instruct",
        "temperature": 0.4,
        "max_tokens": 32768,
        "top_p": 0.95
    },
    "ollama-zephyr (Offline)": {
        "provider": "ollama",
        "model_name": "zephyr:7b-beta",
        "temperature": 0.2,
        "max_tokens": 3072,
        "top_p": 0.9
    },
    "ollama-mixtral (Offline)": {
        "provider": "ollama",
        "model_name": "mixtral:8x7b-instruct",
        "temperature": 0.3,
        "max_tokens": 32768,
        "top_p": 0.9
    },
    "ollama-command-r (Offline)": {
        "provider": "ollama",
        "model_name": "command-r:35b-v0.1",
        "temperature": 0.3,
        "max_tokens": 128000,
        "top_p": 0.9
    },
    "ollama-deepseek (Offline)": {
        "provider": "ollama",
        "model_name": "deepseek-coder:33b-instruct",
        "temperature": 0.2,
        "max_tokens": 16384,
        "top_p": 0.9
    },
    "zephyr-7b-beta (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        "temperature": 0.2,
        "max_new_tokens": 3072,
        "task": "text-generation"
    },
    "llama3-8b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.3,
        "max_new_tokens": 8192,
        "task": "text-generation"
    },
    "llama3-70b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct",
        "temperature": 0.3,
        "max_new_tokens": 8192,
        "task": "text-generation"
    },
    "gpt-4o (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.4,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "gpt-4-turbo (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "temperature": 0.4,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "gpt-3.5-turbo (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo-0125",
        "temperature": 0.5,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "gemini-1.5-flash (Google)": {
        "provider": "google",
        "model_name": "gemini-1.5-flash",
        "temperature": 0.5,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "gemini-1.5-pro (Google)": {
        "provider": "google",
        "model_name": "gemini-1.5-pro",
        "temperature": 0.5,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "llama3-70b (Groq)": {
        "provider": "groq",
        "model_name": "llama3-70b-8192",
        "temperature": 0.3,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "mixtral-8x7b (Groq)": {
        "provider": "groq",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.3,
        "max_tokens": 32768,
        "top_p": 0.9
    },
    "claude-3-haiku (Anthropic)": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "temperature": 0.4,
        "max_tokens": 200000,
        "top_p": 0.9
    },
    "mixtral-8x22b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "temperature": 0.3,
        "max_new_tokens": 65536,
        "task": "text-generation"
    },
    "deepseek-coder-33b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-33b-instruct",
        "temperature": 0.2,
        "max_new_tokens": 16384,
        "task": "text-generation"
    },
    "claude-3-sonnet (Anthropic)": {
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "temperature": 0.4,
        "max_tokens": 200000,
        "top_p": 0.9
    },
    "command-r-plus (Groq)": {
        "provider": "groq",
        "model_name": "command-r-plus",
        "temperature": 0.3,
        "max_tokens": 128000,
        "top_p": 0.9
    },
    "gemini-1.0-pro (Google)": {
        "provider": "google",
        "model_name": "gemini-1.0-pro",
        "temperature": 0.5,
        "max_tokens": 32768,
        "top_p": 0.9
    },
    "phi-3-mini-128k (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-128k-instruct",
        "temperature": 0.3,
        "max_new_tokens": 128000,
        "task": "text-generation"
    }
}

MODEL_USAGE_HINTS = {
    "Select a model": "ℹ️ Please select a model to enable test generation.",
    "ollama-phi (Offline)": "🖥️ Run: `ollama pull phi3:14b-medium-4k-instruct`",
    "ollama-llama3 (Offline)": "🖥️ Run: `ollama pull llama3:70b-instruct`",
    "ollama-mistral (Offline)": "🖥️ Run: `ollama pull mistral:7b-instruct`",
    "ollama-zephyr (Offline)": "🖥️ Run: `ollama pull zephyr:7b-beta`",
    "ollama-mixtral (Offline)": "🖥️ Run: `ollama pull mixtral:8x7b-instruct`",
    "ollama-command-r (Offline)": "🖥️ Run: `ollama pull command-r:35b-v0.1`",
    "ollama-deepseek (Offline)": "🖥️ Run: `ollama pull deepseek-coder:33b-instruct`",
    "zephyr-7b-beta (Hugging Face)": "🌐 Fast 7B model - best for most use cases",
    "llama3-8b (Hugging Face)": "🌐 Efficient 8B model - good balance",
    "llama3-70b (Hugging Face)": "🌐 Meta's flagship model - highest quality",
    "gpt-4o (OpenAI)": "🚀 OpenAI's fastest multimodal model",
    "gpt-4-turbo (OpenAI)": "🧠 OpenAI's most capable model",
    "gpt-3.5-turbo (OpenAI)": "⚡ OpenAI's fastest and most affordable model",
    "gemini-1.5-flash (Google)": "⚡ Google's fastest model - great for testing",
    "gemini-1.5-pro (Google)": "🧠 Google's most capable model",
    "llama3-70b (Groq)": "⚡ Groq - 500+ tokens/sec! Requires GROQ_API_KEY",
    "mixtral-8x7b (Groq)": "⚡ Groq - MoE model at 500+ tokens/sec",
    "claude-3-haiku (Anthropic)": "🏃 Anthropic's fastest model",
    "mixtral-8x22b (Hugging Face)": "🏆 Top open-source model - 176B params",
    "deepseek-coder-33b (Hugging Face)": "💻 Specialist for test case generation",
    "claude-3-sonnet (Anthropic)": "🔍 Best for requirement analysis & edge cases",
    "command-r-plus (Groq)": "🏢 Enterprise-grade RAG optimization",
    "gemini-1.0-pro (Google)": "📊 Most consistent output quality",
    "phi-3-mini-128k (Hugging Face)": "💰 Cost-effective with large 128K context"
}

OLLAMA_TIMEOUT = 600
SKIP_DOMAINS = ["linkedin.com", "facebook.com", "twitter.com", "instagram.com"]

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm(model_name: str):
    if model_name == "Select a model":
        return None
    
    config = MODEL_CONFIG[model_name]
    
    if config["provider"] == "ollama":
        return Ollama(
            model=config["model_name"],
            temperature=config["temperature"],
            num_predict=config["max_tokens"],
            top_p=config["top_p"]
        )
    elif config["provider"] == "openai":
        return ChatOpenAI(
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            model_kwargs={"top_p": config["top_p"]}
        )
    elif config["provider"] == "groq":
        return ChatGroq(
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"]
        )
    elif config["provider"] == "anthropic":
        return ChatAnthropic(
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"]
        )
    elif config["provider"] == "google":
        return ChatGoogleGenerativeAI(
            model=config["model_name"],
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
            top_p=config["top_p"]
        )
    elif config["provider"] == "huggingface":
        return HuggingFaceEndpoint(
            endpoint_url=config["endpoint_url"],
            temperature=config["temperature"],
            max_new_tokens=config["max_new_tokens"],
            model_kwargs={"top_p": config["top_p"]},
            task=config.get("task", "text-generation")
        )
    else:
        raise ValueError(f"Unsupported provider: {config['provider']}")

# ===== UPDATED PROMPT TEMPLATES =====
STANDARD_PROMPT_TEMPLATE = """
As an expert QA engineer, generate concise BDD-style test scenarios:

DOCUMENTATION CONTEXT (Key points only):
{context}

USER REQUIREMENTS (Summary):
{requirements}

REUSABLE TEST STEPS:
{existing_tests}

CURRENT USER STORY:
{current_story}

GENERATION RULES:
- Use strict Gherkin syntax (Background, Scenario, Given/When/Then)
- Do NOT include the "Feature:" keyword - let users add it where needed
- Do NOT use step numbers (1., 2., etc.)
- Create Background for common setup
- Use Examples tables for data variations
- All scenarios must start with "Scenario: " followed by descriptive title
- Use plain language without markdown formatting
- Separate accessibility tests into their own section
- Negative scenarios should ONLY contain negative outcomes
- Positive scenarios should ONLY contain happy paths
"""

TEST_PROMPT = PromptTemplate.from_template(
    STANDARD_PROMPT_TEMPLATE + "\n\nTEST TYPE FOCUS:\n{test_type_instructions}"
)

# Updated test type instructions with strict separation
TEST_TYPE_INSTRUCTIONS = {
    "positive": (
        "Valid inputs → Success outcomes. Include real data examples. "
        "Focus on happy path scenarios. Do NOT include accessibility or error handling. "
        "Do NOT include negative steps. Each scenario must have a clear positive outcome."
    ),
    "negative": (
        "Error conditions → Specific error messages. Cover validation failures. "
        "Include invalid inputs, missing data, and edge cases. "
        "Do NOT include accessibility tests. Do NOT include positive outcomes. "
        "Each scenario must end with an error message or negative outcome."
    ),
    "edge": (
        "Boundary values → Min/Max cases. Temporal/spatial limits. "
        "Cover data boundaries, capacity limits, and extreme conditions. "
        "Do NOT include accessibility tests. Do NOT include positive outcomes."
    ),
    "accessibility": (
        "WCAG 2.1 AA compliance: Keyboard nav, screen readers, contrast, labels. "
        "Format as BDD scenarios. Do NOT include the Feature keyword. "
        "Focus ONLY on accessibility aspects. Do NOT include functional test steps."
    )
}

# ===== CONTEXT FILTERING =====
def filter_test_context(existing_tests: str, test_type: str) -> str:
    """Filter out irrelevant test types from context"""
    filtered_tests = []
    test_blocks = existing_tests.split("// ")
    
    for block in test_blocks:
        if not block.strip():
            continue
            
        # First line is the test type comment
        lines = block.splitlines()
        if not lines:
            continue
            
        block_type = lines[0].strip().lower()
        content = "\n".join(lines[1:])
        
        # Keep only relevant test types
        if test_type == "accessibility":
            if "accessibility" in block_type:
                filtered_tests.append(content)
        else:
            if "accessibility" not in block_type:
                filtered_tests.append(content)
                
    return "\n\n".join(filtered_tests)

async def generate_prompt_async(prompt: str, llm):
    try:
        if isinstance(llm, RunnableLambda):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, llm.invoke, prompt)
        else:
            return await llm.ainvoke(prompt)
    except Exception as e:
        return f"Generation Error: {str(e)}"

async def generate_test_type(test_type, context, requirements, existing_tests, current_story, llm):
    # Filter out irrelevant test types from context
    filtered_tests = filter_test_context(existing_tests, test_type)
    
    full_prompt = TEST_PROMPT.format(
        context=context[:300],  # Reduced context
        requirements=requirements[:200],
        existing_tests=filtered_tests[:400],
        current_story=current_story,
        test_type_instructions=TEST_TYPE_INSTRUCTIONS[test_type]
    )
    return await generate_prompt_async(full_prompt, llm)

async def generate_all_tests(context, requirements, existing_tests, current_story, llm):
    if not llm:
        return ["No model selected"] * 4
    
    test_types = ["positive", "negative", "edge", "accessibility"]
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, test_type in enumerate(test_types):
        status_text.text(f"Generating {test_type} tests...")
        results[test_type] = await generate_test_type(
            test_type, context, requirements, existing_tests, current_story, llm
        )
        progress_bar.progress((i + 1) / len(test_types))
    
    status_text.empty()
    progress_bar.empty()
    return [results[t] for t in test_types]

# ===== IMPROVED CONTEXT HANDLING =====
def get_hybrid_context(vector_store, test_type=None):
    doc_context = ""
    requirements = ""
    existing_tests = ""
    if not vector_store:
        return doc_context, requirements, existing_tests
    
    try:
        # Documentation context
        doc_docs = vector_store.similarity_search(
            "test case generation", 
            k=1,
            filter={"source_type": "documentation"}
        )
        doc_context = "\n".join(doc.page_content[:150] for doc in doc_docs)
        
        # Requirements context
        story_docs = vector_store.similarity_search(
            "user story requirements",
            k=1,
            filter={"source_type": "user_story"}
        )
        requirements = "\n".join(doc.page_content[:150] for doc in story_docs)
        
        # Test context with filtering
        test_docs = vector_store.similarity_search(
            "BDD test cases",
            k=3,
            filter={"source_type": "test_case"}
        )
        
        # Build existing tests with type markers
        test_blocks = []
        for doc in test_docs:
            doc_test_type = doc.metadata.get('test_type', 'test')
            content = doc.page_content[:300]
            test_blocks.append(f"// {doc_test_type}\n{content}")
        
        existing_tests = "\n\n".join(test_blocks)
    except Exception as e:
        st.error(f"Context error: {str(e)}")
    return doc_context, requirements, existing_tests

# ===== HELPER FUNCTIONS =====
def anonymize_story(story: str) -> str:
    story = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+', 'User', story)
    story = re.sub(r'\S+@\S+', 'user@example.com', story)
    story = re.sub(r'\b\d{10}\b', 'XXXXXXXXXX', story)
    return story

def get_story_hash(story: str) -> str:
    return hashlib.sha256(story.encode()).hexdigest()

def story_exists(vector_store, story_hash: str) -> bool:
    if not vector_store:
        return False
    try:
        results = vector_store.get(
            where={"$and": [
                {"source_type": {"$eq": "user_story"}},
                {"story_hash": {"$eq": story_hash}}
            ]}
        )
        return len(results.get('ids', [])) > 0
    except:
        return False

def get_existing_test_cases(vector_store, story_hash: str):
    if not vector_store:
        return None
    try:
        test_cases = {t: "" for t in ["positive", "negative", "edge", "accessibility"]}
        results = vector_store.get(
            where={"$and": [
                {"source_type": {"$eq": "test_case"}},
                {"related_story_hash": {"$eq": story_hash}}
            ]}
        )
        if 'metadatas' in results and 'documents' in results:
            for i, metadata in enumerate(results['metadatas']):
                test_type = metadata.get('test_type', '')
                if test_type in test_cases and i < len(results['documents']):
                    test_cases[test_type] = results['documents'][i]
        return test_cases
    except:
        return None

def url_exists(vector_store, url: str) -> bool:
    if not vector_store:
        return False
    try:
        results = vector_store.get(
            where={"$and": [
                {"source_type": {"$eq": "documentation"}},
                {"source_url": {"$eq": url}}
            ]}
        )
        return len(results.get('ids', [])) > 0
    except:
        return False

# --- Initialize Vector Store ---
embeddings = load_embeddings()

if "vector_store" not in st.session_state:
    if os.path.exists("./chroma_db"):
        try:
            st.session_state.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# --- UI Implementation ---
# Initialize session state variables
if "generating" not in st.session_state:
    st.session_state.generating = False
if "raw_tests" not in st.session_state:
    st.session_state.raw_tests = {}
if "edited_tests" not in st.session_state:
    st.session_state.edited_tests = {
        "positive": "", "negative": "", "edge": "", "accessibility": ""
    }
# Initialize RAG context storage
if "rag_context" not in st.session_state:
    st.session_state.rag_context = {
        "documentation": "",
        "requirements": "",
        "existing_tests": ""
    }

with st.sidebar:
    st.header("⚙️ Configuration")
    model_names = list(MODEL_CONFIG.keys())
    default_index = model_names.index("gpt-4o (OpenAI)") if "gpt-4o (OpenAI)" in model_names else 0
    model_name = st.selectbox("Select Model", model_names, index=default_index, key="model_selector")
    
    # Show speed warning
    if model_name != "Select a model":
        model_speed = MODEL_CONFIG[model_name].get("speed", "medium")
        if model_speed in ["slow", "very slow"]:
            st.warning(f"⚠️ This model is {model_speed.replace('_', ' ')}, consider cloud options")
    
    st.caption(MODEL_USAGE_HINTS.get(model_name, ""))

    st.subheader("🔗 Documentation URLs")
    url_input = st.text_area("Enter one or more URLs:",
                             value="https://example.com/features/",
                             key="url_input")
    
    # Skip heavy JS sites
    skip_heavy_sites = st.checkbox("⏩ Skip complex sites (social media)", value=True,
                                  help="Improves processing speed")
    
    store_stories = st.checkbox("📚 Store user stories for context", value=True)
    
    # Vector DB stats
    if "vector_store" in st.session_state and st.session_state.vector_store:
        try:
            collection = st.session_state.vector_store.get()
            user_story_count = len([m for m in collection['metadatas'] if m.get("source_type") == "user_story"])
            test_case_count = len([m for m in collection['metadatas'] if m.get("source_type") == "test_case"])
            doc_count = len([m for m in collection['metadatas'] if m.get("source_type") == "documentation"])
            
            st.metric("Stored User Stories", user_story_count)
            st.metric("Stored Test Cases", test_case_count)
            st.metric("Stored Documents", doc_count)
        except Exception as e:
            st.error(f"Error retrieving vector DB stats: {str(e)}")
    
    if st.button("Process URL(s)", disabled=st.session_state.generating):
        with st.spinner("🔄 Loading URLs..."):
            docs = []
            new_url_count = 0
            skipped_count = 0
            
            for url in url_input.splitlines():
                clean_url = url.strip()
                if not clean_url:
                    continue
                
                # Skip existing URLs
                if st.session_state.get("vector_store") and url_exists(st.session_state.vector_store, clean_url):
                    st.info(f"⏩ Already processed: {clean_url}")
                    continue
                    
                # Skip heavy JS sites
                if skip_heavy_sites and any(domain in clean_url for domain in SKIP_DOMAINS):
                    skipped_count += 1
                    st.info(f"⏭️ Skipped complex site: {clean_url}")
                    continue
                    
                try:
                    loader = SeleniumURLLoader(
                        urls=[clean_url],
                        continue_on_failure=True
                    )
                    loaded_docs = loader.load()
                    timestamp = datetime.datetime.now().isoformat()
                    for doc in loaded_docs:
                        doc.metadata.update({
                            "source_url": clean_url,
                            "ingested_at": timestamp,
                            "source_type": "documentation",
                        })
                    docs.extend(loaded_docs)
                    new_url_count += 1
                    st.success(f"✅ Loaded: {clean_url}")
                except Exception as e:
                    st.error(f"❌ Failed to load {clean_url}: {str(e)}")
            
            if docs:
                if "vector_store" not in st.session_state or not st.session_state.vector_store:
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=docs, 
                        embedding=embeddings,
                        persist_directory="./chroma_db"
                    )
                else:
                    st.session_state.vector_store.add_documents(docs)
                
                msg = f"📚 Processed {new_url_count} new URLs ({len(docs)} chunks)"
                if skipped_count:
                    msg += f", skipped {skipped_count} complex sites"
                st.success(msg)

with st.form("input_form"):
    st.subheader("🧾 User Story")
    user_story = st.text_area("Paste user story + acceptance criteria", 
                             height=200, 
                             placeholder="As a user, I want to... so that I can...",
                             key="user_story")
    
    # Show warning for long stories
    if user_story:
        if len(user_story) > 1000:
            st.warning("⚠️ Long user stories may slow down generation. Consider summarizing.")
    
    # Check if story exists
    if user_story and "vector_store" in st.session_state and st.session_state.vector_store:
        anonymized = anonymize_story(user_story)
        story_hash = get_story_hash(anonymized)
        if story_exists(st.session_state.vector_store, story_hash):
            st.warning("⚠️ This user story already exists in the database")
    
    submitted = st.form_submit_button(
        "🚀 Generate Test Cases",
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
                
                # Check for existing tests first
                anonymized = anonymize_story(user_story)
                story_hash = get_story_hash(anonymized)
                existing_tests = None
                
                if "vector_store" in st.session_state and st.session_state.vector_store:
                    existing_tests = get_existing_test_cases(
                        st.session_state.vector_store, 
                        story_hash
                    )
                
                if existing_tests and sum(1 for t in existing_tests.values() if t.strip()) >= 3:
                    st.session_state.edited_tests = existing_tests
                    st.session_state.raw_tests = existing_tests
                    st.toast("✅ Loaded existing test cases", icon="💾")
                    elapsed = time.perf_counter() - start_time
                    minutes = elapsed // 60
                    seconds = elapsed % 60
                    if minutes > 0:
                        st.info(f"⏱️ Completed in {minutes:.0f}m {seconds:.1f}s")
                    else:
                        st.info(f"⏱️ Completed in {seconds:.1f}s")
                else:
                    with st.spinner("🧠 Generating test cases..."):
                        llm = get_llm(model_name)
                        
                        # Store user story if enabled
                        if store_stories and len(user_story) > 50:
                            try:
                                if not story_exists(st.session_state.vector_store, story_hash):
                                    story_doc = Document(
                                        page_content=anonymized,
                                        metadata={
                                            "source_type": "user_story",
                                            "ingested_at": datetime.datetime.now().isoformat(),
                                            "story_hash": story_hash
                                        }
                                    )
                                    if "vector_store" not in st.session_state:
                                        st.session_state.vector_store = Chroma.from_documents(
                                            [story_doc], 
                                            embeddings,
                                            persist_directory="./chroma_db"
                                        )
                                    else:
                                        st.session_state.vector_store.add_documents([story_doc])
                            except Exception as e:
                                st.error(f"Error storing user story: {str(e)}")
                        
                        # Get and store RAG context
                        if st.session_state.get("vector_store"):
                            doc_context, requirements, existing_test_steps = get_hybrid_context(
                                st.session_state.vector_store
                            )
                        else:
                            doc_context, requirements, existing_test_steps = "", "", ""
                        
                        st.session_state.rag_context = {
                            "documentation": doc_context,
                            "requirements": requirements,
                            "existing_tests": existing_test_steps
                        }
                        
                        # Generate tests
                        results = asyncio.run(
                            generate_all_tests(doc_context, requirements, existing_test_steps, user_story, llm)
                        )
                        positive, negative, edge, accessibility = results
                        
                        # Store results
                        st.session_state.raw_tests = {
                            "positive": positive,
                            "negative": negative,
                            "edge": edge,
                            "accessibility": accessibility
                        }
                        st.session_state.edited_tests = {
                            "positive": positive,
                            "negative": negative,
                            "edge": edge,
                            "accessibility": accessibility
                        }
                        
                        elapsed = time.perf_counter() - start_time
                        minutes = elapsed // 60
                        seconds = elapsed % 60
                        if minutes > 0:
                            st.info(f"⏱️ Generated in {minutes:.0f}m {seconds:.1f}s")
                        else:
                            st.info(f"⏱️ Generated in {seconds:.1f}s")
        finally:
            st.session_state.generating = False

# Display test cases and RAG context
if "raw_tests" in st.session_state and st.session_state.raw_tests:
    st.divider()
    
    # Show RAG context in expander section
    with st.expander("🧠 RAG Context Used for Generation", expanded=False):
        st.subheader("Documentation Context")
        doc_context = st.session_state.rag_context["documentation"]
        if doc_context:
            st.info(doc_context)
        else:
            st.warning("No documentation context available")
            
        st.subheader("Requirements Context")
        req_context = st.session_state.rag_context["requirements"]
        if req_context:
            st.info(req_context)
        else:
            st.warning("No requirements context available")
            
        st.subheader("Existing Test Context")
        test_context = st.session_state.rag_context["existing_tests"]
        if test_context:
            st.info(test_context)
        else:
            st.warning("No existing test context available")
    
    st.subheader("✏️ Edit and Save Test Cases")
    
    tabs = st.tabs(["✅ Positive", "❌ Negative", "🟧 Edge", "♿ Accessibility"])
    test_types = ["positive", "negative", "edge", "accessibility"]
    
    for i, test_type in enumerate(test_types):
        with tabs[i]:
            st.session_state.edited_tests[test_type] = st.text_area(
                f"Edit {test_type} scenarios", 
                value=st.session_state.edited_tests[test_type],
                height=300,
                key=f"edit_{test_type}"
            )
            
            # Validation for test types
            if test_type == "positive":
                if "accessibility" in st.session_state.edited_tests[test_type].lower():
                    st.warning("⚠️ Positive tests should not contain accessibility scenarios")
                if "Feature:" in st.session_state.edited_tests[test_type]:
                    st.info("ℹ️ Remember to add Feature keyword where needed")
                    
            if test_type == "negative" and "Then I see the" in st.session_state.edited_tests[test_type]:
                st.warning("⚠️ Negative tests should not contain positive outcomes")
            
            if test_type == "accessibility" and "Feature:" in st.session_state.edited_tests[test_type]:
                st.info("ℹ️ Remember to add Feature keyword where needed")
            
            # Dedicated save button for each test type
            if st.button(f"💾 Save {test_type.capitalize()} Tests", 
                         key=f"save_{test_type}",
                         disabled=st.session_state.generating):
                if "vector_store" in st.session_state and st.session_state.vector_store:
                    test_docs = []
                    anonymized = anonymize_story(user_story)
                    story_hash = get_story_hash(anonymized)
                    
                    content = st.session_state.edited_tests[test_type]
                    if content.strip():
                        test_docs.append(Document(
                            page_content=content,
                            metadata={
                                "source_type": "test_case",
                                "test_type": test_type,
                                "related_story": anonymized,
                                "related_story_hash": story_hash,
                                "created_at": datetime.datetime.now().isoformat()
                            }
                        ))
                    
                    if test_docs:
                        try:
                            st.session_state.vector_store.add_documents(test_docs)
                            st.success(f"✅ {test_type.capitalize()} tests saved to vector DB!")
                        except Exception as e:
                            st.error(f"Error saving tests: {str(e)}")
                    else:
                        st.warning("No content to save")
                else:
                    st.warning("Vector store not initialized. Process URLs first.")