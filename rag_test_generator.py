from collections import defaultdict
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
        "model_name": "llama3.1:8b",
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

STANDARD_PROMPT_TEMPLATE = """
You are an expert QA engineer.
Your task is to generate concise BDD-style test scenarios for the given user story and context.
IMPORTANT:
- Your output MUST contain ONLY the Gherkin scenarios.
- Do NOT include any explanation, markdown formatting, headings, code fences, or any extra text.
- Each scenario must start with: Scenario: <title>
- Use plain text only.
- You MUST incorporate information retrieved from the vector database knowledge base. If relevant information exists in the context sections below, USE it to inform your scenarios.
- Do NOT invent features, functionality, or scenarios that are not supported by the provided context or user story.
- Only generate test scenarios for the specific test type provided below.
DOCUMENTATION CONTEXT:
{context}
USER REQUIREMENTS:
{requirements}
REUSABLE TEST STEPS:
{existing_tests}
CURRENT USER STORY:
{current_story}
TEST TYPE:
{test_type_instructions}
"""

TEST_PROMPT = PromptTemplate.from_template(STANDARD_PROMPT_TEMPLATE)

TEST_TYPE_INSTRUCTIONS = {
    "positive": (
        "Valid inputs → Success outcomes. Include realistic, meaningful examples. "
        "Focus only on happy path scenarios. Do NOT include accessibility, edge, or negative paths. "
        "Each scenario must represent a clear, complete success case."
    ),
    "negative": (
        "Invalid inputs → Specific error messages. Include validation errors, business rule violations, missing or malformed data. "
        "Do NOT include accessibility or success outcomes. "
        "Each scenario must represent a unique and valid failure path."
    ),
    "edge": (
        "Extreme and boundary conditions → Max/Min inputs, empty values, large payloads, timeout, or unusual sequences. "
        "Do NOT include accessibility or generic negative scenarios. "
        "Each case should challenge system stability or limits."
    ),
    "accessibility": (
        "Accessibility criteria based on WCAG 2.1 AA: Keyboard navigation, screen reader labels, contrast ratio, alt text, ARIA. "
        "Represent each accessibility check as a separate scenario. "
        "Do NOT include functional or validation steps. Focus only on a11y requirements."
    )
}

def filter_test_context(existing_tests: str, test_type: str) -> str:
    filtered_tests = []
    test_blocks = existing_tests.split("// ")
    for block in test_blocks:
        if not block.strip():
            continue
        lines = block.splitlines()
        if not lines:
            continue
        block_type = lines[0].strip().lower()
        content = "\n".join(lines[1:])
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
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, llm.invoke, prompt)
        else:
            return await llm.ainvoke(prompt)
    except Exception as e:
        return f"Generation Error: {str(e)}"

async def generate_test_type(test_type, context, requirements, existing_tests, current_story, llm, custom_instruction=None):
    filtered_tests = filter_test_context(existing_tests, test_type)
    instruction = custom_instruction if custom_instruction else TEST_TYPE_INSTRUCTIONS[test_type]
    full_prompt = TEST_PROMPT.format(
        context=context[:300],
        requirements=requirements[:200],
        existing_tests=filtered_tests[:400],
        current_story=current_story,
        test_type_instructions=instruction
    )
    return await generate_prompt_async(full_prompt, llm)

def get_hybrid_context(vector_store, test_type=None):
    doc_context = ""
    requirements = ""
    existing_tests = ""
    if not vector_store:
        return doc_context, requirements, existing_tests
    try:
        doc_docs = vector_store.similarity_search(
            "test case generation", 
            k=1,
            filter={"source_type": "documentation"}
        )
        doc_context = "\n".join(doc.page_content[:150] for doc in doc_docs)
        story_docs = vector_store.similarity_search(
            "user story requirements",
            k=1,
            filter={"source_type": "user_story"}
        )
        requirements = "\n".join(doc.page_content[:150] for doc in story_docs)
        test_docs = vector_store.similarity_search(
            "BDD test cases",
            k=3,
            filter={"source_type": "test_case"}
        )
        test_blocks = []
        for doc in test_docs:
            doc_test_type = doc.metadata.get('test_type', 'test')
            content = doc.page_content[:300]
            test_blocks.append(f"// {doc_test_type}\n{content}")
        existing_tests = "\n\n".join(test_blocks)
    except Exception as e:
        st.error(f"Context error: {str(e)}")
    return doc_context, requirements, existing_tests

def anonymize_story(story: str) -> str:
    import re
    story = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+', 'User', story)
    story = re.sub(r'\S+@\S+', 'user@example.com', story)
    story = re.sub(r'\b\d{10}\b', 'XXXXXXXXXX', story)
    return story

def get_story_hash(story: str) -> str:
    import hashlib
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
        return {t: "" for t in ["positive", "negative", "edge", "accessibility"]}
    try:
        test_cases = {t: [] for t in ["positive", "negative", "edge", "accessibility"]}
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
                    test_cases[test_type].append((results['documents'][i], results['ids'][i]))
        return {t: "\n\n".join(doc for doc, _ in test_cases[t]) for t in test_cases}
    except Exception as e:
        st.error(f"Error reading existing test cases: {e}")
        return {t: "" for t in ["positive", "negative", "edge", "accessibility"]}

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

def split_scenarios(test_text):
    scenarios = re.split(r"(?=Scenario:)", test_text, flags=re.MULTILINE)
    return [s.strip() for s in scenarios if s.strip()]

def delete_tests_of_type(vector_store, story_hash, test_type):
    if not vector_store:
        return
    results = vector_store.get(
        where={"$and": [
            {"source_type": {"$eq": "test_case"}},
            {"related_story_hash": {"$eq": story_hash}},
            {"test_type": {"$eq": test_type}}
        ]}
    )
    ids = results.get("ids", [])
    if ids:
        vector_store.delete(ids)

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

if "generating" not in st.session_state:
    st.session_state.generating = False
if "raw_tests" not in st.session_state:
    st.session_state.raw_tests = {}
if "edited_tests" not in st.session_state:
    st.session_state.edited_tests = {
        "positive": "", "negative": "", "edge": "", "accessibility": ""
    }
if "rag_context" not in st.session_state:
    st.session_state.rag_context = {
        "documentation": "",
        "requirements": "",
        "existing_tests": ""
    }
if "show_section" not in st.session_state:
    st.session_state.show_section = None

with st.sidebar:
    st.header("⚙️ Configuration")
    model_names = list(MODEL_CONFIG.keys())
    default_index = model_names.index("gpt-4o (OpenAI)") if "gpt-4o (OpenAI)" in model_names else 0
    model_name = st.selectbox("Select Model", model_names, index=default_index, key="model_selector")
    st.subheader("🔗 Documentation URLs")
    url_input = st.text_area("Enter one or more URLs:",
                             value="https://example.com/features/",
                             key="url_input")
    skip_heavy_sites = st.checkbox("⏩ Skip complex sites (social media)", value=True, help="Improves processing speed")
    store_stories = st.checkbox("📚 Store user stories for context", value=True)
    process_url_clicked = st.button("Process URL(s)", disabled=st.session_state.generating)
    st.divider()
    if "vector_store" in st.session_state and st.session_state.vector_store:
        collection = st.session_state.vector_store.get()
        user_story_count = len([m for m in collection['metadatas'] if m.get("source_type") == "user_story"])
        test_case_count = len([m for m in collection['metadatas'] if m.get("source_type") == "test_case"])
        doc_count = len([m for m in collection['metadatas'] if m.get("source_type") == "documentation"])
        if st.button(f"📖 Show User Stories ({user_story_count})"):
            st.session_state.show_section = "stories"
        if st.button(f"📝 Show Test Cases ({test_case_count})"):
            st.session_state.show_section = "tests"
        if st.button(f"📄 Show Documents ({doc_count})"):
            st.session_state.show_section = "docs"

    if process_url_clicked:
        with st.spinner("🔄 Loading URLs..."):
            docs = []
            new_url_count = 0
            skipped_count = 0
            for url in url_input.splitlines():
                clean_url = url.strip()
                if not clean_url:
                    continue
                if st.session_state.get("vector_store") and url_exists(st.session_state.vector_store, clean_url):
                    st.info(f"⏩ Already processed: {clean_url}")
                    continue
                if skip_heavy_sites and any(domain in clean_url for domain in SKIP_DOMAINS):
                    skipped_count += 1
                    st.info(f"⏭️ Skipped complex site: {clean_url}")
                    continue
                try:
                    loader = SeleniumURLLoader(urls=[clean_url], continue_on_failure=True)
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
    submitted = st.form_submit_button(
        "🚀 Generate Test Cases",
        disabled=st.session_state.generating
    )
    if submitted:
        st.session_state.generating = True
        if not user_story.strip():
            st.warning("⚠️ Please enter a user story.")
            st.session_state.generating = False
        elif model_name == "Select a model":
            st.warning("⚠️ Please select a model to proceed.")
            st.session_state.generating = False
        else:
            anonymized = anonymize_story(user_story)
            story_hash = get_story_hash(anonymized)
            if store_stories and len(user_story) > 50 and not story_exists(st.session_state.vector_store, story_hash):
                story_doc = Document(
                    page_content=anonymized,
                    metadata={
                        "source_type": "user_story",
                        "ingested_at": datetime.datetime.now().isoformat(),
                        "story_hash": story_hash
                    }
                )
                if not st.session_state.vector_store:
                    st.session_state.vector_store = Chroma.from_documents(
                        [story_doc], 
                        embeddings,
                        persist_directory="./chroma_db"
                    )
                else:
                    st.session_state.vector_store.add_documents([story_doc])
            doc_context, requirements, existing_test_steps = get_hybrid_context(
                st.session_state.vector_store
            )
            llm = get_llm(model_name)
            stored = get_existing_test_cases(st.session_state.vector_store, story_hash)
            new_tests = {}
            import asyncio
            for test_type in ["positive", "negative", "edge", "accessibility"]:
                if stored[test_type]:
                    new_tests[test_type] = stored[test_type]
                else:
                    with st.spinner(f"Generating {test_type} tests..."):
                        result = asyncio.run(
                            generate_test_type(
                                test_type, doc_context, requirements, existing_test_steps, user_story, llm
                            )
                        )
                        new_tests[test_type] = result
            st.session_state.raw_tests = dict(new_tests)
            st.session_state.edited_tests = dict(new_tests)
            st.session_state.rag_context = {
                "documentation": doc_context,
                "requirements": requirements,
                "existing_tests": existing_test_steps
            }
            st.session_state.generating = False

if st.session_state.get("raw_tests"):
    st.divider()
    with st.expander("🧠 RAG Context Used for Generation", expanded=False):
        st.subheader("Documentation Context")
        doc_context = st.session_state.rag_context.get("documentation", "")
        if doc_context:
            st.info(doc_context)
        else:
            st.warning("No documentation context available")
        st.subheader("Requirements Context")
        req_context = st.session_state.rag_context.get("requirements", "")
        if req_context:
            st.info(req_context)
        else:
            st.warning("No requirements context available")
        st.subheader("Existing Test Context")
        test_context = st.session_state.rag_context.get("existing_tests", "")
        if test_context:
            st.info(test_context)
        else:
            st.warning("No existing test context available")
    st.subheader("✏️ Edit, Regenerate and Save Test Cases")
    test_types = ["positive", "negative", "edge", "accessibility"]
    tabs = st.tabs([f"✅ Positive", "❌ Negative", "🟧 Edge", "♿ Accessibility"])
    for i, test_type in enumerate(test_types):
        with tabs[i]:
            ta_key = f"edit_{test_type}_textarea"
            st.session_state.edited_tests[test_type] = st.text_area(
                f"{test_type.capitalize()} Scenarios",
                value=st.session_state.edited_tests[test_type],
                height=300,
                key=ta_key
            )
            custom_prompt = st.text_area(
                f"Add extra instruction for {test_type} (optional, for Regenerate)",
                value="",
                key=f"custom_prompt_{test_type}"
            )
            regen_key = f"regen_{test_type}_btn"
            if st.button(f"Regenerate {test_type.capitalize()} Tests", key=regen_key):
                with st.spinner(f"Regenerating {test_type} tests..."):
                    doc_context, requirements, existing_test_steps = get_hybrid_context(
                        st.session_state.vector_store
                    )
                    llm = get_llm(model_name)
                    import asyncio
                    result = asyncio.run(
                        generate_test_type(
                            test_type, doc_context, requirements, existing_test_steps, user_story, llm, custom_instruction=custom_prompt
                        )
                    )
                    current_val = st.session_state.edited_tests.get(test_type, "")
                    if current_val.strip():
                        new_val = current_val.strip() + "\n\n" + result.strip()
                    else:
                        new_val = result.strip()
                    st.session_state.edited_tests[test_type] = new_val
                    st.success(f"Regenerated {test_type} tests. Edit before saving if needed.")
                    st.rerun()
            save_key = f"save_{test_type}_btn"
            if st.button(f"💾 Save {test_type.capitalize()} Tests", key=save_key):
                anonymized = anonymize_story(user_story)
                story_hash = get_story_hash(anonymized)
                delete_tests_of_type(st.session_state.vector_store, story_hash, test_type)
                content = st.session_state.edited_tests[test_type]
                scenarios = split_scenarios(content)
                test_docs = []
                for scenario_text in scenarios:
                    if scenario_text:
                        test_docs.append(Document(
                            page_content=scenario_text,
                            metadata={
                                "source_type": "test_case",
                                "test_type": test_type,
                                "related_story": anonymized,
                                "related_story_hash": story_hash,
                                "created_at": datetime.datetime.now().isoformat()
                            }
                        ))
                if test_docs:
                    st.session_state.vector_store.add_documents(test_docs)
                    st.success(f"Saved {len(test_docs)} {test_type} test(s) to vector DB!")
                else:
                    st.warning("No scenarios to save.")
                st.rerun()

# === SHOW THE CHOSEN SECTION (User Stories / Test Cases / Documents) ===
if "vector_store" in st.session_state and st.session_state.vector_store and st.session_state.get("show_section"):
    results = st.session_state.vector_store.get()
    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])
    ids = results.get("ids", [])

    st.divider()
    if st.session_state.show_section == "stories":
        st.markdown("### 📚 Stored User Stories")
        stories = [d for m, d in zip(metadatas, documents) if m.get("source_type") == "user_story"]
        if stories:
            for doc in stories:
                st.code(doc.strip(), language=None)
        else:
            st.info("No user stories found.")

    elif st.session_state.show_section == "tests":
        st.markdown("### 📝 Stored Test Cases")
        test_cases = [(i, m.get("test_type", ""), d) for i, (m, d) in enumerate(zip(metadatas, documents)) if m.get("source_type") == "test_case"]
        test_types = ["positive", "negative", "edge", "accessibility"]
        tabs = st.tabs([t.capitalize() for t in test_types])

        # CSS for perfect top-alignment of Save/Delete buttons
        st.markdown("""
            <style>
            .test-scenario-area textarea {
                margin-top: 0.2em;
                margin-bottom: 0.2em;
                font-size: 1rem;
            }
            /* Column for buttons - always top-aligned */
            div[data-testid="column"]:nth-of-type(2) {
                display: flex !important;
                flex-direction: column !important;
                align-items: flex-end !important;
                justify-content: flex-start !important;
                height: 100%;
            }
            div[data-testid="column"]:nth-of-type(2) button {
                margin-top: 0px !important;
                margin-bottom: 10px !important;
                min-width: 32px !important;
                max-width: 32px !important;
                min-height: 32px !important;
                max-height: 32px !important;
                font-size: 1.1rem !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            </style>
        """, unsafe_allow_html=True)

        for idx_type, t in enumerate(test_types):
            with tabs[idx_type]:
                scenarios = [(i, doc) for i, typ, doc in test_cases if typ == t]
                if scenarios:
                    for i, s in scenarios:
                        col_main, col_btns = st.columns([8, 1], gap="medium")
                        textarea_key = f"edit_textarea_{t}_{i}"
                        with col_main:
                            with st.container():
                                st.markdown('<div class="test-scenario-area">', unsafe_allow_html=True)
                                updated_text = st.text_area(
                                    label="",
                                    value=s,
                                    key=textarea_key,
                                    height=120
                                )
                                st.markdown('</div>', unsafe_allow_html=True)
                        with col_btns:
                            save_btn_key = f"save_{t}_{i}_btn"
                            delete_btn_key = f"delete_{t}_{i}_btn"
                            save_clicked = st.button("💾", key=save_btn_key)
                            delete_clicked = st.button("🗑️", key=delete_btn_key)
                            if save_clicked:
                                doc_id = ids[i]
                                st.session_state.vector_store.delete([doc_id])
                                m = metadatas[i]
                                test_type = m.get('test_type', '')
                                related_story = m.get('related_story', '')
                                related_story_hash = m.get('related_story_hash', '')
                                new_doc = Document(
                                    page_content=updated_text,
                                    metadata={
                                        "source_type": "test_case",
                                        "test_type": test_type,
                                        "related_story": related_story,
                                        "related_story_hash": related_story_hash,
                                        "created_at": datetime.datetime.now().isoformat()
                                    }
                                )
                                st.session_state.vector_store.add_documents([new_doc])
                                st.success("Test case updated!")
                                st.rerun()
                            if delete_clicked:
                                doc_id = ids[i]
                                st.session_state.vector_store.delete([doc_id])
                                st.success("Test case deleted!")
                                st.rerun()
                else:
                    st.info(f"No {t} test cases found.")

    elif st.session_state.show_section == "docs":
        st.markdown("### 📄 Stored Documents")
        docs = [(m.get("source_url", ""), d) for m, d in zip(metadatas, documents) if m.get("source_type") == "documentation"]
        if docs:
            for url, doc in docs:
                st.markdown(f"**URL**: [{url}]({url})")
                st.markdown(f"<div style='font-size:smaller'>{doc}</div>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No documents found.")
            
