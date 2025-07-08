### -------------------- IMPORTS & DEPENDENCIES -------------------- ###

from collections import defaultdict
from pathlib import Path
from langchain_core.prompts import PromptTemplate
import streamlit as st
import asyncio
import datetime
import time
import re
import hashlib
import os
import numpy as np
import warnings
import json

# LangChain, Chroma, Embeddings, Document
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

# Import additional model providers (OpenAI, Groq, Anthropic, Google, Ollama)
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_ollama import Ollama
except ImportError:
    from langchain_community.llms import Ollama

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

### -------------------- MODEL/LLM CONFIGURATION -------------------- ###

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
    elif config["provider"] == "groq":
        return ChatGroq(
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            model_kwargs={"top_p": config["top_p"]}
        )
    elif config["provider"] == "anthropic":
        return ChatAnthropic(
            model=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
    elif config["provider"] == "google":
        return ChatGoogleGenerativeAI(
            model=config["model_name"],
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
            top_p=config["top_p"]
        )
    else:
        raise ValueError(f"Unsupported provider: {config['provider']}")

### -------------------- EMBEDDINGS AND VECTOR STORE SETUP -------------------- ###

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

OLLAMA_TIMEOUT = 600

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

from langchain_core.prompts import PromptTemplate

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
RECORDER CONTEXT:
{recorder_context}
REUSABLE TEST STEPS:
{existing_tests}
CURRENT USER STORY:
{current_story}
TEST TYPE:
{test_type_instructions}
"""

TEST_PROMPT = PromptTemplate.from_template(STANDARD_PROMPT_TEMPLATE)

def get_vector_store(embeddings, persist_dir="./chroma_db"):
    """Safely load or create the Chroma vector store."""
    if os.path.exists(persist_dir):
        try:
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None
    else:
        return None

### -------------------- HELPER FUNCTIONS (GENERIC) -------------------- ###
def get_existing_test_cases(vector_store, story_hash: str):
    """
    Returns a dict of test type -> joined scenarios for the provided story_hash,
    using only test cases linked to that user story.
    """
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
                    test_cases[test_type].append(results['documents'][i])
        return {t: "\n\n".join(test_cases[t]) for t in test_cases}
    except Exception as e:
        st.error(f"Error reading existing test cases: {e}")
        return {t: "" for t in ["positive", "negative", "edge", "accessibility"]}

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

def anonymize_story(story: str) -> str:
    story = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+', 'User', story)
    story = re.sub(r'\S+@\S+', 'user@example.com', story)
    story = re.sub(r'\b\d{10}\b', 'XXXXXXXXXX', story)
    return story

def get_story_hash(story: str) -> str:
    return hashlib.sha256(story.encode()).hexdigest()

def split_scenarios(test_text):
    scenarios = re.split(r"(?=Scenario:)", test_text, flags=re.MULTILINE)
    return [s.strip() for s in scenarios if s.strip()]

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

def filter_test_context(existing_tests: str, test_type: str) -> str:
    """Filter test steps by type (for context reuse)."""
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

def get_hybrid_context(vector_store, with_recorder=False):
    """
    Retrieves hybrid RAG context (documentation, requirements, existing tests, optionally recorder steps)
    from the vector store, for use in LLM prompts.
    Returns: doc_context, requirements, existing_tests, recorder_context
    """
    doc_context = ""
    requirements = ""
    existing_tests = ""
    recorder_context = ""
    if not vector_store:
        return doc_context, requirements, existing_tests, recorder_context

    try:
        # Get closest documentation chunk(s)
        doc_docs = vector_store.similarity_search(
            "test case generation", 
            k=1,
            filter={"source_type": "documentation"}
        )
        doc_context = "\n".join(doc.page_content[:150] for doc in doc_docs)
        
        # Get closest user story (requirements) chunk(s)
        story_docs = vector_store.similarity_search(
            "user story requirements",
            k=1,
            filter={"source_type": "user_story"}
        )
        requirements = "\n".join(doc.page_content[:150] for doc in story_docs)
        
        # Get most relevant test cases (existing tests)
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
        
        # Optionally: Get latest or most relevant recorder (user action) context
        if with_recorder:
            rec_docs = vector_store.similarity_search(
                "user action steps",
                k=1,
                filter={"source_type": "recorder"}
            )
            if rec_docs:
                recorder_context = rec_docs[0].page_content[:1000]
    except Exception as e:
        st.error(f"Context error: {str(e)}")

    return doc_context, requirements, existing_tests, recorder_context

def generate_test_type(test_type, context, requirements, existing_tests, recorder_context, current_story, llm, custom_instruction=None):
    filtered_tests = filter_test_context(existing_tests, test_type)
    instruction = custom_instruction if custom_instruction else TEST_TYPE_INSTRUCTIONS[test_type]
    full_prompt = TEST_PROMPT.format(
        context=context[:300],
        requirements=requirements[:200],
        recorder_context=recorder_context[:500],
        existing_tests=filtered_tests[:400],
        current_story=current_story,
        test_type_instructions=instruction
    )
    if hasattr(llm, "invoke"):
        return llm.invoke(full_prompt)
    elif hasattr(llm, "ainvoke"):
        import asyncio
        return asyncio.run(llm.ainvoke(full_prompt))
    else:
        raise Exception("LLM object does not support invoke/ainvoke")

def detect_existing_framework(path: Path, language: str, tool: str, style: str) -> bool:
    """
    Returns True if the requested framework already exists on disk.
    """
    lang = language.lower()
    tl  = tool.lower()
    stl = style.lower()

    # Playwright + TS + BDD
    if tl == "playwright" and lang == "typescript" and stl == "bdd":
        return (path / "features").exists() and (path / "steps").exists()

    # Playwright + TS + Plain (non-BDD)
    if tl == "playwright" and lang == "typescript" and stl != "bdd":
        return (path / "tests" / "e2e").exists() and (path / "pages").exists()

    # Existing logic for other langs/tools:
    if lang == "python":
        return (path / "requirements.txt").exists()
    if lang in ("javascript", "typescript"):
        return (path / "package.json").exists()
    if lang == "java":
        return (path / "pom.xml").exists() or (path / "build.gradle").exists()
    if lang == "c#":
        return any(path.glob("*.csproj"))

    return False

def scaffold_framework(path: Path, language: str, tool: str, style: str):
    """
    Create on-disk folders/files for the chosen framework.
    """
    lang = language.lower()
    tl  = tool.lower()
    stl = style.lower()

    path.mkdir(parents=True, exist_ok=True)

    # Playwright + TS + BDD
    if tl == "playwright" and lang == "typescript" and stl == "bdd":
        # 1) features/, 2) steps/, 3) supports/pages/, 4) config/, 5) env/
        for d in ("features", "steps", "supports/pages", "config", "env"):
            (path / d).mkdir(parents=True, exist_ok=True)
        # stub config file
        (path / "config" / "playwright.config.ts").write_text("""
import { defineConfig } from '@playwright/test';
export default defineConfig({
  use: { baseURL: process.env.BASE_URL },
});
""".strip())
        # basic package.json + tsconfig.json + README.md
        (path / "package.json").write_text("""
{
  "name": "playwright-bdd-ts-pom",
  "devDependencies": { "@playwright/test": "^1.0.0" }
}
""".strip())
        (path / "tsconfig.json").write_text("""
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "commonjs",
    "outDir": "dist"
  }
}
""".strip())
        (path / "README.md").write_text("# Playwright BDD TypeScript POM")

        return

    # Playwright + TS + non-BDD
    if tl == "playwright" and lang == "typescript" and stl != "bdd":
        for d in ("tests/e2e", "pages", "utils", "config", "env"):
            (path / d).mkdir(parents=True, exist_ok=True)
        (path / "config" / "playwright.config.ts").write_text("""
import { defineConfig } from '@playwright/test';
export default defineConfig({});
""".strip())
        (path / "package.json").write_text("""
{
  "name": "playwright-ts-pom",
  "devDependencies": { "@playwright/test": "^1.0.0" }
}
""".strip())
        (path / "README.md").write_text("# Playwright TypeScript POM")
        return

    # Fallback: minimal skeleton
    if lang == "python":
        (path / "requirements.txt").write_text("pytest\nselenium\n")
        (path / "tests").mkdir(exist_ok=True)
    elif lang in ("javascript", "typescript"):
        (path / "package.json").write_text('{ "name": "auto-project", "devDependencies": {} }')
        (path / "tests").mkdir(exist_ok=True)
    elif lang == "java":
        (path / "pom.xml").write_text("<project/>")
        (path / "src" / "test" / "java").mkdir(parents=True, exist_ok=True)
    elif lang == "c#":
        (path / "auto_project.csproj").write_text("<Project/>")
        (path / "Tests").mkdir(exist_ok=True)

def get_test_directory(path: Path, language: str) -> Path:
    if language.lower() == "python":
        return path / "tests"
    if language.lower() in ["javascript", "typescript"]:
        return path / "tests"
    if language.lower() == "java":
        return path / "src" / "test" / "java"
    if language.lower() == "c#":
        return path / "Tests"
    return path / "tests"


def get_latest_recorder_context(vector_store):
    """Return the most recently stored recorder file content as string, or None."""
    recorder_docs = []
    if vector_store:
        results = vector_store.get()
        for m, doc in zip(results.get("metadatas", []), results.get("documents", [])):
            if m.get("source_type") == "recorder":
                recorder_docs.append((m.get("ingested_at", ""), doc))
    if recorder_docs:
        recorder_docs.sort(reverse=True)
        return recorder_docs[0][1]
    return None

def generate_test_script_with_llm(
    llm, language, tool, style, scenario, test_type=None,  
    app_url=None, recorder_context=None, doc_context=None, requirements=None
):
    """
    LLM-based script generation for a single scenario.
    Produces automation code in strict scalable POM/project structure,
    based on user selections for tool, language, style, and test_type.
    """
    folder_layout = ""
    extra_rules = ""
    file_naming_hint = ""
    
    if tool.lower() == "playwright" and language.lower() == "typescript":
        if style.lower() == "bdd":
            folder_layout = """
Strictly use this project structure for Playwright-BDD TypeScript:
playwright-bdd-ts-pom/
├── features/
│   └── <feature_name>.feature
├── steps/
│   └── <feature_name>.steps.ts
├── supports/
│   └── pages/
│       ├── basePage.ts
│       └── <pageName>.ts
├── config/
│   └── playwright.config.ts
├── env/
│   └── .env files
├── package.json, tsconfig.json, README.md, etc.
""".strip()
            extra_rules = """
- Place all step definitions in `/steps/` as `<feature_name>.steps.ts`.
- Page Object Model classes go in `/supports/pages/`.
- Feature files go in `/features/` as `<feature_name>.feature`.
- For each test type (positive/negative/edge/accessibility), create a separate `.feature` file (e.g., `positive.feature`).
- Show code for BOTH the page object(s) and step definition(s) for the given scenario and test type.
""".strip()
            file_naming_hint = f"File naming: features/{test_type}.feature, steps/{test_type}.steps.ts"
        else:
            folder_layout = """
Strictly use this project structure for Playwright TypeScript (non-BDD):
playwright-ts-pom/
├── tests/e2e/
│   └── <feature>.spec.ts
├── pages/
│   ├── basePage.ts
│   └── <pageName>.ts
├── utils/
├── config/
│   └── playwright.config.ts
├── env/
│   └── .env files
├── package.json, tsconfig.json, README.md, etc.
""".strip()
            extra_rules = """
- UI test specs in `/tests/e2e/` as `<feature>.spec.ts`.
- POM classes in `/pages/`.
- Helper/utils in `/utils/`.
- Test data/fixtures in `/tests/fixtures/`.
- For each test type (positive/negative/edge/accessibility), create a separate spec file (e.g., `positive.spec.ts`).
- Show code for BOTH the page object(s) and the corresponding test spec for the given scenario and test type.
""".strip()
            file_naming_hint = f"File naming: tests/e2e/{test_type}.spec.ts"
    else:
        # Fallback for other tools/languages: instruct to follow their conventions
        folder_layout = f"Use industry-standard folder structure for {tool} and {language}. Only code—no markdown/explanations."
        extra_rules = ""
        file_naming_hint = f"(Test type: {test_type})"

    # --- CONTEXT: doc, requirements, URL, recorder ---
    doc_context_snippet = f"\nDOCUMENTATION CONTEXT:\n{doc_context.strip()}" if doc_context else ""
    requirements_snippet = f"\nREQUIREMENTS/USER STORY:\n{requirements.strip()}" if requirements else ""
    recorder_snippet = f"\nRECORDER STEPS:\n{recorder_context.strip()}" if recorder_context else ""
    url_line = f"\nBASE URL: {app_url}" if app_url else "\nBASE URL: NO_URL"

    # === FINAL PROMPT ===
    prompt = f"""
You are an expert SDET.
Generate ONLY valid, scalable, production-grade automation code for Playwright in TypeScript ({style} style), using the structure below.

{folder_layout}
{extra_rules}
{file_naming_hint}

{doc_context_snippet}
{requirements_snippet}
{url_line}
{recorder_snippet}

SCENARIO:
{scenario}

STRICT RULES:
- Do NOT include any markdown, explanation, or extra text—only pure code.
- Follow strict POM, modular, and scalable best practices.
""".strip()

    if hasattr(llm, "invoke"):
        return llm.invoke(prompt)
    elif hasattr(llm, "ainvoke"):
        import asyncio
        return asyncio.run(llm.ainvoke(prompt))
    else:
        raise Exception("LLM object does not support invoke/ainvoke")



def get_latest_recorder_context(vector_store):
    """Return the most recently stored recorder file content as string, or None."""
    recorder_docs = []
    if vector_store:
        results = vector_store.get()
        for m, doc in zip(results.get("metadatas", []), results.get("documents", [])):
            if m.get("source_type") == "recorder":
                recorder_docs.append((m.get("ingested_at", ""), doc))
    if recorder_docs:
        recorder_docs.sort(reverse=True)
        return recorder_docs[0][1]
    return None
def visualize_connections(vector_store):
    import pandas as pd
    table = []
    if vector_store:
        results = vector_store.get()
        for m, doc in zip(results.get("metadatas", []), results.get("documents", [])):
            if m.get("source_type") == "test_case":
                table.append({
                    "Story Hash": m.get("related_story_hash", ""),
                    "Test Type": m.get("test_type", ""),
                    "Test Case": doc[:60] + "...",
                    "Recorder Linked": m.get("recorder_file", ""),
                    "Doc/URL": m.get("source_url", "")
                })
    if table:
        st.markdown("### 🕸️ Test Story/Test/Recorder/URL Connections")
        st.dataframe(pd.DataFrame(table))
    else:
        st.info("No connections to display yet.")
def show_errors(errors):
    if errors:
        for e in errors:
            st.error(e)

def feedback_for_deduplication(skipped_count):
    if skipped_count > 0:
        st.sidebar.info(f"Skipped {skipped_count} duplicate scenarios/scripts.")
### -------------------- UI CONFIGURATION & STATE -------------------- ###

st.set_page_config(page_title="Test Case Generator", layout="wide")
st.title("🧪 AI Test Case Generator")

if "vector_store" not in st.session_state:
    embeddings = load_embeddings()
    st.session_state.vector_store = get_vector_store(embeddings)
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
        "existing_tests": "",
        "recorder": ""
    }
if "show_section" not in st.session_state:
    st.session_state.show_section = None
if "stored_recorder_files" not in st.session_state:
    st.session_state.stored_recorder_files = []
with st.sidebar:
    st.header("⚙️ Configuration")
    model_names = list(MODEL_CONFIG.keys())
    default_index = model_names.index("gpt-4o (OpenAI)") if "gpt-4o (OpenAI)" in model_names else 0
    model_name = st.selectbox("Select Model", model_names, index=default_index, key="model_selector")
    st.subheader("🔗 Application/Feature URLs")
    url_input = st.text_area("Enter one or more URLs (one per line):",
                             value="https://example.com/features/",
                             key="url_input")
    store_stories = st.checkbox("📚 Store user stories for context", value=True)
    process_url_clicked = st.button("Process URL(s)", disabled=st.session_state.generating)

    # Recorder JSON upload
    st.subheader("📝 Upload DevTools Recorder (JSON)")
    recorder_json_file = st.file_uploader("Upload Chrome DevTools Recorder JSON", type="json", key="recorder_json")
    process_recorder_clicked = st.button("Process Recorder JSON", disabled=st.session_state.generating)

    # Vector store statistics and section toggles
    if st.session_state.vector_store:
        collection = st.session_state.vector_store.get()
        user_story_count = len([m for m in collection['metadatas'] if m.get("source_type") == "user_story"])
        test_case_count = len([m for m in collection['metadatas'] if m.get("source_type") == "test_case"])
        doc_count = len([m for m in collection['metadatas'] if m.get("source_type") == "documentation"])
        rec_count = len([m for m in collection['metadatas'] if m.get("source_type") == "recorder"])
        if st.button(f"📖 Show User Stories ({user_story_count})"):
            st.session_state.show_section = "stories"
        if st.button(f"📝 Show Test Cases ({test_case_count})"):
            st.session_state.show_section = "tests"
        if st.button(f"📄 Show Documents ({doc_count})"):
            st.session_state.show_section = "docs"
        if st.button(f"🎬 Show Recorder Steps ({rec_count})"):
            st.session_state.show_section = "recorders"
# --- Process URLs for documentation/context ingestion ---
if process_url_clicked:
    with st.spinner("🔄 Loading URLs..."):
        docs = []
        new_url_count = 0
        for url in url_input.splitlines():
            clean_url = url.strip()
            if not clean_url:
                continue
            if st.session_state.vector_store and url_exists(st.session_state.vector_store, clean_url):
                st.info(f"⏩ Already processed: {clean_url}")
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
            if not st.session_state.vector_store:
                st.session_state.vector_store = Chroma.from_documents(
                    documents=docs,
                    embedding=load_embeddings(),
                    persist_directory="./chroma_db"
                )
            else:
                st.session_state.vector_store.add_documents(docs)
            st.success(f"📚 Processed {new_url_count} new URLs ({len(docs)} chunks)")

# --- Process DevTools Recorder JSON upload ---
if process_recorder_clicked and recorder_json_file:
    try:
        recorder_json = json.load(recorder_json_file)
        # Store a summary of steps as a vector DB document for retrieval
        steps_text = json.dumps(recorder_json, indent=2)  # Store full for now; can be summarized
        timestamp = datetime.datetime.now().isoformat()
        doc = Document(
            page_content=steps_text[:10000],  # chunk/limit as needed
            metadata={
                "source_type": "recorder",
                "ingested_at": timestamp,
                "file_name": recorder_json_file.name
            }
        )
        if not st.session_state.vector_store:
            st.session_state.vector_store = Chroma.from_documents(
                [doc], load_embeddings(), persist_directory="./chroma_db"
            )
        else:
            st.session_state.vector_store.add_documents([doc])
        st.session_state.stored_recorder_files.append(recorder_json_file.name)
        st.success(f"Recorder file '{recorder_json_file.name}' ingested.")
    except Exception as e:
        st.error(f"❌ Recorder JSON parse failed: {e}")
with st.form("input_form"):
    st.subheader("🧾 User Story")
    user_story = st.text_area(
        "Paste user story + acceptance criteria",
        height=200,
        placeholder="As a user, I want to... so that I can...",
        key="user_story"
    )
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
                        load_embeddings(),
                        persist_directory="./chroma_db"
                    )
                else:
                    st.session_state.vector_store.add_documents([story_doc])
            # Gather context (docs, requirements, recorder, tests)
            doc_context, requirements, existing_test_steps, recorder_context = get_hybrid_context(
                st.session_state.vector_store, with_recorder=True
            )
            llm = get_llm(model_name)
            stored = get_existing_test_cases(st.session_state.vector_store, story_hash)
            new_tests = {}
            for test_type in ["positive", "negative", "edge", "accessibility"]:
                if stored[test_type]:
                    new_tests[test_type] = stored[test_type]
                else:
                    with st.spinner(f"Generating {test_type} tests..."):
                        result = generate_test_type(
                                test_type, doc_context, requirements, existing_test_steps, recorder_context, user_story, llm
                            )
                        
                        new_tests[test_type] = result
            st.session_state.raw_tests = dict(new_tests)
            st.session_state.edited_tests = dict(new_tests)
            st.session_state.rag_context = {
                "documentation": doc_context,
                "requirements": requirements,
                "existing_tests": existing_test_steps,
                "recorder": recorder_context
            }
            st.session_state.generating = False
if st.session_state.get("raw_tests"):
    st.divider()
    with st.expander("🧠 RAG Context Used for Generation", expanded=False):
        st.subheader("Documentation Context")
        doc_context = st.session_state.rag_context.get("documentation", "")
        st.info(doc_context or "No documentation context available")
        st.subheader("Requirements Context")
        req_context = st.session_state.rag_context.get("requirements", "")
        st.info(req_context or "No requirements context available")
        st.subheader("Recorder Context")
        rec_context = st.session_state.rag_context.get("recorder", "")
        st.info(rec_context or "No recorder context available")
        st.subheader("Existing Test Context")
        test_context = st.session_state.rag_context.get("existing_tests", "")
        st.info(test_context or "No existing test context available")
if st.session_state.get("raw_tests"):
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
                    doc_context, requirements, existing_test_steps, recorder_context = get_hybrid_context(
                        st.session_state.vector_store, with_recorder=True
                    )
                    llm = get_llm(model_name)
                    result = generate_test_type(
                            test_type, doc_context, requirements, existing_test_steps, recorder_context, user_story, llm, custom_instruction=custom_prompt
                            )
                    
                    current_val = st.session_state.edited_tests.get(test_type, "")
                    new_val = (current_val.strip() + "\n\n" + result.strip()) if current_val.strip() else result.strip()
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
        for idx_type, t in enumerate(test_types):
            with tabs[idx_type]:
                scenarios = [(i, doc) for i, typ, doc in test_cases if typ == t]
                if scenarios:
                    for i, s in scenarios:
                        col_main, col_btns = st.columns([8, 1], gap="medium")
                        textarea_key = f"edit_textarea_{t}_{i}"
                        with col_main:
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

    elif st.session_state.show_section == "recorders":
        st.markdown("### 🎬 Stored Recorder Files & Steps")
        recorder_files = [(m.get("file_name", f"Recorder_{i}.json"), d) for i, (m, d) in enumerate(zip(metadatas, documents)) if m.get("source_type") == "recorder"]
        if recorder_files:
            for file_name, doc in recorder_files:
                st.markdown(f"**Recorder File**: {file_name}")
                st.markdown(f"<pre style='font-size:smaller'>{doc}</pre>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No recorder files found.")

# Sidebar controls at bottom of file
st.sidebar.markdown("### 🔧 Generate Automation Scripts from Stored Test Cases")
selected_tool  = st.sidebar.selectbox("Tool", ["Playwright", "Selenium"])
selected_lang  = st.sidebar.selectbox(
    "Language", ["Python", "JavaScript", "TypeScript", "Java", "C#"]
)
selected_style = st.sidebar.radio("Style", ["Plain", "BDD"], horizontal=True)
target_path    = st.sidebar.text_input(
    "Output Path", value=str(Path.cwd() / "auto_project")
)

# Rebuild URL list for selecting base URL
vector_store = st.session_state.get("vector_store")
urls = []
if vector_store:
    coll = vector_store.get()
    for m in coll["metadatas"]:
        if m.get("source_type") == "documentation" and m.get("source_url"):
            urls.append(m["source_url"])
selected_app_url = st.sidebar.selectbox(
    "Select Application URL for Scripts", urls, key="script_app_url_selectbox"
)


if st.sidebar.button("Generate Automation Scripts (Stored Only)"):
    with st.spinner("🚀 Generating automation scripts..."):

        # 1) Prepare output directory & scaffold if needed
        output_dir = Path(target_path)
        if not detect_existing_framework(output_dir, selected_lang, selected_tool, selected_style):
            scaffold_framework(output_dir, selected_lang, selected_tool, selected_style)
            st.sidebar.info("Scaffolded new automation framework.")

        # 2) Load LLM & RAG context
        llm = get_llm(st.session_state["model_selector"])
        doc_ctx, reqs, _, recorder_ctx = get_hybrid_context(
            st.session_state.vector_store, with_recorder=True
        )

        # 3) Group stored test_case docs by test_type
        results = st.session_state.vector_store.get()
        grouped = defaultdict(list)
        for m, doc in zip(results["metadatas"], results["documents"]):
            if m.get("source_type") == "test_case":
                tt = m.get("test_type", "positive")
                grouped[tt].append(doc)

        tl  = selected_tool.lower()
        lang = selected_lang.lower()
        stl = selected_style.lower()

        # --- PLAYWRIGHT + TS + BDD -----------------------------------------------
        if tl == "playwright" and lang == "typescript" and stl == "bdd":
            feat_dir = output_dir / "features"; feat_dir.mkdir(parents=True, exist_ok=True)
            step_dir = output_dir / "steps";     step_dir.mkdir(parents=True, exist_ok=True)
            pom_dir  = output_dir / "supports" / "pages"; pom_dir.mkdir(parents=True, exist_ok=True)

            for test_type, scenarios in grouped.items():
                if not scenarios:
                    continue

                # join scenarios once
                scenario_block = "\n\n".join(scenarios)

                # --- Generate Feature File ---
                feature_prompt = (
                    "You are an expert QA engineer.\n"
                    "Generate ONLY the Gherkin feature file (no fences, no markdown) for these scenarios:\n\n"
                    f"{scenario_block}"
                )
                feature_code = llm.invoke(feature_prompt).strip()
                (feat_dir / f"{test_type}.feature").write_text(feature_code, encoding="utf-8")

                # --- Generate Steps + POM Files ---
                step_prompt = (
                    "You are a senior SDET.\n"
                    "Generate ONLY Playwright-BDD TypeScript step definitions and Page Object Model classes.\n\n"
                    "For each file, prefix with exactly:\n"
                    "===FILE: <relative-path>===\n"
                    "<typescript code>\n\n"
                    "- steps/{test_type}.steps.ts should contain all Given/When/Then implementations.\n"
                    "- supports/pages/BasePage.ts must be included.\n"
                    "- Any additional page classes you reference go under supports/pages/.\n\n"
                    "Do NOT include the feature itself, comments, or any explanations.\n\n"
                    f"{scenario_block}"
                )
                step_response = llm.invoke(step_prompt).strip()

                # Split out each file and write
                for part in re.split(r"===FILE:", step_response):
                    part = part.strip()
                    if not part:
                        continue
                    header, code = part.split("===", 1)
                    rel_path = header.strip()
                    file_path = output_dir / rel_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(code.strip(), encoding="utf-8")

        # --- PLAYWRIGHT + TS + non-BDD -------------------------------------------
        elif tl == "playwright" and lang == "typescript":
            test_dir = output_dir / "tests" / "e2e"
            test_dir.mkdir(parents=True, exist_ok=True)

            for test_type, scenarios in grouped.items():
                if not scenarios:
                    continue

                scenario_block = "\n\n".join(scenarios)
                spec_prompt = (
                    "You are a senior SDET.\n"
                    "Generate ONLY a Playwright TypeScript test spec (no markdown) for these scenarios:\n\n"
                    f"{scenario_block}"
                )
                code = llm.invoke(spec_prompt).strip()
                (test_dir / f"{test_type}.spec.ts").write_text(code, encoding="utf-8")

        # --- FALLBACK: Python / JavaScript / Java / C# ---------------------------
        else:
            test_dir = get_test_directory(output_dir, selected_lang)
            test_dir.mkdir(parents=True, exist_ok=True)

            ext_map = {
                "python": "py",
                "javascript": "js",
                "typescript": "ts",
                "java": "java",
                "c#": "cs",
            }
            ext = ext_map.get(lang, "txt")

            for test_type, scenarios in grouped.items():
                if not scenarios:
                    continue

                scenario_block = "\n\n".join(scenarios)
                fallback_prompt = (
                    "You are a senior SDET.\n"
                    f"Generate ONLY a {selected_tool} {selected_lang} test script (no markdown) for these scenarios:\n\n"
                    f"{scenario_block}"
                )
                code = llm.invoke(fallback_prompt).strip()
                (test_dir / f"{test_type}_test.{ext}").write_text(code, encoding="utf-8")

        st.sidebar.success(f"Generated/updated scripts in {output_dir}")





