### -------------------- IMPORTS & DEPENDENCIES -------------------- ###

from collections import defaultdict
import logging
from pathlib import Path
import textwrap
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
        "temperature": 0.3,  # Balanced for structured output
        "max_tokens": 4096,  # Suitable for smaller model and single scenarios
        "top_p": 0.85,  # Ensures coherence for code-like output
    },
    "ollama-llama3 (Offline)": {
        "provider": "ollama",
        "model_name": "llama3.1:8b",
        "temperature": 0.3,  # Balanced for structured output
        "max_tokens": 3072,  # Supports multiple scenarios/scripts
        "top_p": 0.85,  # Ensures coherence
    },
    "ollama-mistral (Offline)": {
        "provider": "ollama",
        "model_name": "mistral:7b-instruct",
        "temperature": 0.3,  # Reduced for precision
        "max_tokens": 8192,  # Reduced to reasonable limit for test generation
        "top_p": 0.85,  # Reduced for coherence
    },
    "ollama-deepseek-coder (Offline)": {
        "provider": "ollama",
        "model_name": "deepseek-coder:6.7b-instruct",
        "temperature": 0.2,  # Increased for flexibility, optimized for code generation
        "max_tokens": 3072,  # Increased from 5000 to handle multiple scenarios
        "top_p": 0.85,  # Reduced for higher coherence
    },
    "ollama-codellama (Offline)": {
        "provider": "ollama",
        "model_name": "codellama:34b-instruct",
        "temperature": 0.3,  # Increased for flexibility, optimized for code
        "max_tokens": 8192,  # Increased from 5000 to handle multiple scenarios
        "top_p": 0.85,  # Reduced for coherence
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
        "temperature": 0.3,  # Balanced for structured output
        "max_tokens": 16384,  # Reduced to reasonable limit for complex outputs
        "top_p": 0.85,  # Adjusted for coherence
    },
    "ollama-command-r (Offline)": {
        "provider": "ollama",
        "model_name": "command-r:35b-v0.1",
        "temperature": 0.3,  # Balanced for structured output
        "max_tokens": 16384,  # Reduced from 128000 to optimize resource use
        "top_p": 0.85,  # Adjusted for coherence
    },
    "ollama-deepseek (Offline)": {
        "provider": "ollama",
        "model_name": "deepseek-coder:33b-instruct",
        "temperature": 0.3,  # Adjusted for consistency and flexibility
        "max_tokens": 16384,  # Matches larger models for complex outputs
        "top_p": 0.85,  # Adjusted for coherence
    },
     "ollama-codellama:70b-instruct (Offline)": {
        "provider": "ollama",
        "model_name": "codellama:70b-instruct",
        "temperature": 0.3,
        "max_tokens": 8192,
        "top_p": 0.85
    },
    "ollama-codegemma:7b-instruct (Offline)": {
        "provider": "ollama",
        "model_name": "codegemma:7b-instruct",
        "temperature": 0.2,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "ollama-codestral:latest (Offline)": {
        "provider": "ollama",
        "model_name": "codestral:latest",
        "temperature": 0.25,
        "max_tokens": 16384,
        "top_p": 0.85
    },
    "ollama-codeqwen:7b (Offline)": {
        "provider": "ollama",
        "model_name": "codeqwen:7b",
        "temperature": 0.3,
        "max_tokens": 16384,
        "top_p": 0.85
    },
    "ollama-codegeex:4b (Offline)": {
        "provider": "ollama",
        "model_name": "codegeex:4b",
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.9
    },
    "ollama-deepseek-coder-v2:16b (Offline)": {
        "provider": "ollama",
        "model_name": "deepseek-coder-v2:16b",
        "temperature": 0.2,
        "max_tokens": 16384,
        "top_p": 0.85
    },
    "ollama-qwen2.5-coder:7b (Offline)": {
        "provider": "ollama",
        "model_name": "qwen2.5-coder:7b",
        "temperature": 0.3,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "zephyr-7b-beta (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        "temperature": 0.0,  # Adjusted for precision
        "max_new_tokens": 4096,  # Increased for consistency with smaller models
        "top_p": 0.85,  # Added for coherence
        "task": "text-generation",
    },
    "llama3-8b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        "temperature": 0.3,  # Consistent with other Llama models
        "max_new_tokens": 8192,  # Matches ollama-llama3
        "top_p": 0.85,  # Added for coherence
        "task": "text-generation",
    },
    "llama3-70b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct",
        "temperature": 0.3,  # Consistent with other Llama models
        "max_new_tokens": 8192,  # Matches ollama-llama3
        "top_p": 0.85,  # Added for coherence
        "task": "text-generation",
    },
    "gpt-4o (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.4,  # Kept for balanced natural language generation
        "max_tokens": 4096,  # Suitable for high-quality model
        "top_p": 0.9,  # Kept for slight diversity in natural language
    },
    "gpt-4-turbo (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "temperature": 0.4,  # Kept for balanced natural language generation
        "max_tokens": 4096,  # Suitable for high-quality model
        "top_p": 0.9,  # Kept for slight diversity
    },
    "gpt-3.5-turbo (OpenAI)": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo-0125",
        "temperature": 0.4,  # Reduced for more precision
        "max_tokens": 4096,  # Suitable for efficient model
        "top_p": 0.9,  # Kept for slight diversity
    },
    "gemini-1.5-flash (Google)": {
        "provider": "google",
        "model_name": "gemini-1.5-flash",
        "temperature": 0.4,  # Reduced for precision
        "max_tokens": 8192,  # Matches mid-sized models
        "top_p": 0.9,  # Kept for slight diversity
    },
    "gemini-1.5-pro (Google)": {
        "provider": "google",
        "model_name": "gemini-1.5-pro",
        "temperature": 0.4,  # Reduced for precision
        "max_tokens": 8192,  # Matches mid-sized models
        "top_p": 0.9,  # Kept for slight diversity
    },
    "llama3-70b (Groq)": {
        "provider": "groq",
        "model_name": "llama3-70b-8192",
        "temperature": 0.3,  # Consistent with other Llama models
        "max_tokens": 8192,  # Matches model capability
        "top_p": 0.85,  # Adjusted for coherence
    },
    "mixtral-8x7b (Groq)": {
        "provider": "groq",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.3,  # Consistent with ollama-mixtral
        "max_tokens": 16384,  # Reduced to optimize resource use
        "top_p": 0.85,  # Adjusted for coherence
    },
    "claude-3-haiku (Anthropic)": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "temperature": 0.4,  # Kept for balanced natural language
        "max_tokens": 16384,  # Reduced from 200000 to optimize
        "top_p": 0.9,  # Kept for slight diversity
    },
    "mixtral-8x22b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "temperature": 0.3,  # Consistent with other Mixtral models
        "max_new_tokens": 16384,  # Reduced to optimize
        "top_p": 0.85,  # Added for coherence
        "task": "text-generation",
    },
    "deepseek-coder-33b (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-33b-instruct",
        "temperature": 0.3,  # Consistent with ollama-deepseek
        "max_new_tokens": 16384,  # Matches ollama-deepseek
        "top_p": 0.85,  # Added for coherence
        "task": "text-generation",
    },
    "claude-3-sonnet (Anthropic)": {
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "temperature": 0.4,  # Kept for balanced natural language
        "max_tokens": 16384,  # Reduced from 200000 to optimize
        "top_p": 0.9,  # Kept for slight diversity
    },
    "command-r-plus (Groq)": {
        "provider": "groq",
        "model_name": "command-r-plus",
        "temperature": 0.3,  # Consistent with ollama-command-r
        "max_tokens": 16384,  # Reduced from 128000 to optimize
        "top_p": 0.85,  # Adjusted for coherence
    },
    "gemini-1.0-pro (Google)": {
        "provider": "google",
        "model_name": "gemini-1.0-pro",
        "temperature": 0.4,  # Reduced for precision
        "max_tokens": 8192,  # Reduced to match mid-sized models
        "top_p": 0.9,  # Kept for slight diversity
    },
    "phi-3-mini-128k (Hugging Face)": {
        "provider": "huggingface",
        "endpoint_url": "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-128k-instruct",
        "temperature": 0.3,  # Consistent with ollama-phi
        "max_new_tokens": 16384,  # Reduced to optimize for smaller model
        "top_p": 0.85,  # Added for coherence
        "task": "text-generation",
    },
}

MODEL_USAGE_HINTS = {
    "Select a model": "ℹ️ Please select a model to enable test generation.",
    "ollama-codellama:70b-instruct (Offline)": "🖥️ Run: `ollama pull codellama:70b-instruct` - Meta's best coding model",
    "ollama-codegemma:7b-instruct (Offline)": "🖥️ Run: `ollama pull codegemma:7b-instruct` - Google's lightweight coding model",
    "ollama-codestral:latest (Offline)": "🖥️ Run: `ollama pull codestral` - Mistral's cutting-edge coding assistant",
    "ollama-codeqwen:7b (Offline)": "🖥️ Run: `ollama pull codeqwen:7b` - Alibaba's Qwen coding model",
    "ollama-codegeex:4b (Offline)": "🖥️ Run: `ollama pull codegeex:4b` - Tencent's multi-language coding model",
    "ollama-deepseek-coder-v2:16b (Offline)": "🖥️ Run: `ollama run deepseek-coder-v2:16b` - DeepSeek's latest coding model",
    "ollama-qwen2.5-coder:7b (Offline)": "🖥️ Run: `ollama pull qwen2.5-coder:7b` - Alibaba's newest coding model",
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

OLLAMA_TIMEOUT = 5000

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
- **Always** include a `STEP DEFINITIONS:` section for {test_type} tests, even if they mirror other cases.
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

def get_hybrid_context(vector_store, with_recorder=False, recorder_limit=3, doc_limit=1, story_limit=1, test_limit=3):
    """
    Retrieves enhanced RAG context including documentation, requirements, existing tests, and recorder data.

    Args:
        vector_store: Chroma vector store instance.
        with_recorder: whether to include recorder context.
        recorder_limit: number of recorder documents to retrieve.
        doc_limit: number of documentation chunks to retrieve.
        story_limit: number of user story chunks to retrieve.
        test_limit: number of test_case chunks to retrieve.

    Returns:
        doc_context (str), requirements (str), existing_tests (str), recorder_context (str)
    """
    doc_context = ""
    requirements = ""
    existing_tests = ""
    recorder_context = ""
    if not vector_store:
        return doc_context, requirements, existing_tests, recorder_context

    try:
        # Documentation: Top-K most relevant chunks
        docs = vector_store.similarity_search(
            "documentation context for test generation",
            k=doc_limit,
            filter={"source_type": "documentation"}
        )
        doc_context = "\n".join(doc.page_content for doc in docs)

        # User story requirements
        stories = vector_store.similarity_search(
            "user story requirements",
            k=story_limit,
            filter={"source_type": "user_story"}
        )
        requirements = "\n".join(st.page_content for st in stories)

        # Existing test cases
        tests = vector_store.similarity_search(
            "existing BDD test cases",
            k=test_limit,
            filter={"source_type": "test_case"}
        )
        blocks = []
        for doc in tests:
            ttype = doc.metadata.get('test_type', 'test')
            content = doc.page_content
            blocks.append(f"// {ttype}\n{content}")
        existing_tests = "\n\n".join(blocks)

        # Recorder context: full JSON and extracted selectors
        if with_recorder:
            rec_docs = vector_store.similarity_search(
                "devtools recorder steps",
                k=recorder_limit,
                filter={"source_type": "recorder"}
            )
            raw_jsons = [doc.page_content for doc in rec_docs]
            # Combine raw JSON dumps
            combined = "\n--- RECORDER JSON ---\n".join(raw_jsons)
            # Parse each JSON to extract selector mappings if possible
            try:
                import json
                selectors = []
                for j in raw_jsons:
                    data = json.loads(j)
                    mapping = extract_selectors_from_recorder(data)
                    selectors.append(json.dumps(mapping, indent=2))
                selector_block = "\n--- SELECTOR MAPPING ---\n".join(selectors)
                recorder_context = f"{combined}\n{selector_block}"
            except Exception:
                recorder_context = combined
    except Exception as e:
        logger.error(f"Error building hybrid context: {e}")

    return doc_context, requirements, existing_tests, recorder_context


def generate_test_type(test_type, context, requirements, existing_tests, recorder_context, current_story, llm, custom_instruction=None):
    filtered_tests = filter_test_context(existing_tests, test_type)
    instruction = custom_instruction if custom_instruction else TEST_TYPE_INSTRUCTIONS[test_type]
    full_prompt = TEST_PROMPT.format(
        context=context[:10000],
        requirements=requirements[:10000],
        recorder_context=recorder_context[:10000],
        existing_tests=filtered_tests[:10000],
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
    tl = tool.lower()
    stl = style.lower()

    path.mkdir(parents=True, exist_ok=True)

    # Playwright + TS + BDD
    if tl == "playwright" and lang == "typescript" and stl == "bdd":
        for d in ("features", "steps", "supports/pages", "config", "env"):
            (path / d).mkdir(parents=True, exist_ok=True)
        (path / "config" / "playwright.config.ts").write_text("""
import { defineConfig } from '@playwright/test';
export default defineConfig({
  use: { baseURL: process.env.BASE_URL },
});
""".strip())
        (path / "package.json").write_text("""
{
  "name": "playwright-bdd-ts-pom",
  "devDependencies": {
    "@playwright/test": "^1.47.0",
    "@cucumber/cucumber": "^10.0.0",
    "@axe-core/playwright": "^4.9.0",
    "typescript": "^5.5.0"
  },
  "scripts": {
    "test": "cucumber-js"
  }
}
""".strip())
        (path / "tsconfig.json").write_text("""
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "commonjs",
    "outDir": "dist",
    "strict": true,
    "esModuleInterop": true
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
  "devDependencies": {
    "@playwright/test": "^1.47.0",
    "typescript": "^5.5.0"
  }
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

def remove_duplicate_imports(content: str) -> str:
    """
    Given the full text of a TypeScript/JS file, extract all import statements,
    dedupe & sort them, and move them to the very top of the file.
    """
    lines = content.splitlines()
    import_lines: List[str] = []
    other_lines: List[str] = []

    # Separate imports from everything else
    for line in lines:
        if line.strip().startswith("import "):
            # Normalize spacing/trailing semicolon
            cleaned = line.strip().rstrip(";")
            import_lines.append(cleaned + ";")
        else:
            other_lines.append(line)

    # Dedupe & sort
    unique_imports = sorted(set(import_lines))

    # Reassemble: imports block + a blank line + the rest
    cleaned_content = "\n".join(unique_imports)
    rest = "\n".join(other_lines).lstrip("\n")  # drop leading blank lines

    return f"{cleaned_content}\n\n{rest}"

def extract_imports_and_steps(content: str) -> tuple:
    """Extract imports and step definitions from step file content"""
    import_lines = []
    step_defs = []
    
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") and "from" in stripped:
            import_lines.append(line)
        elif stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
            step_defs.append(line)
    
    return "\n".join(import_lines), "\n".join(step_defs)

### Add parse_llm_output function ###
def parse_llm_output(raw_output: str, test_type: str, step_ext: str, pom_ext: str) -> list:
    """
    Enhanced parser that properly handles step definitions and POM classes
    with robust marker detection and fallback mechanisms
    """
    files = []
    current_file = {"path": "", "content": ""}
    lines = raw_output.splitlines()
    i = 0
    
    # Track found files to avoid duplicates
    found_files = set()
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 1. Detect explicit markers
        if line.upper().startswith("FEATURE FILE:"):
            if current_file["path"]:
                files.append(current_file)
            current_file = {"path": f"features/{test_type}.feature", "content": ""}
            found_files.add("feature")
            i += 1
            continue
            
        if line.upper().startswith("STEP DEFINITIONS:"):
            if current_file["path"]:
                files.append(current_file)
            current_file = {"path": f"steps/{test_type}.{step_ext}", "content": ""}
            found_files.add("steps")
            i += 1
            continue
            
        if line.upper().startswith("POM CLASS:"):
            if current_file["path"]:
                files.append(current_file)
            # Extract class name from next line if not on same line
            if " " in line:
                cls_name = line.split(" ", 1)[1].strip()
            else:
                i += 1
                cls_name = lines[i].strip() if i < len(lines) else "Page"
            pom_path = f"supports/pages/{cls_name}.{pom_ext}"
            current_file = {"path": pom_path, "content": ""}
            found_files.add(pom_path)
            i += 1
            continue
            
        # 2. Fallback detection for Gherkin
        if not current_file["path"] and line.startswith("Feature:"):
            if current_file["path"]:
                files.append(current_file)
            current_file = {"path": f"features/{test_type}.feature", "content": line + "\n"}
            found_files.add("feature")
            i += 1
            continue
            
        # 3. Fallback for step definitions (look for import patterns)
        if not current_file["path"] and ("import" in line and ("Given" in line or "When" in line or "Then" in line)):
            if current_file["path"]:
                files.append(current_file)
            current_file = {"path": f"steps/{test_type}.{step_ext}", "content": line + "\n"}
            found_files.add("steps")
            i += 1
            continue
            
        # 4. Fallback for POM classes (look for class/export patterns)
        if not current_file["path"] and ("class " in line or "export " in line) and "page" in line.lower():
            if current_file["path"]:
                files.append(current_file)
            # Try to extract class name
            cls_match = re.search(r'(class|export\s+default\s+class)\s+(\w+)', line)
            cls_name = cls_match.group(2) if cls_match else "Page"
            pom_path = f"supports/pages/{cls_name}.{pom_ext}"
            current_file = {"path": pom_path, "content": line + "\n"}
            found_files.add(pom_path)
            i += 1
            continue
            
        # 5. Accumulate content for current file
        if current_file["path"]:
            current_file["content"] += lines[i] + "\n"
        
        i += 1

    # Add last file
    if current_file["path"] and current_file["content"].strip():
        files.append(current_file)

    # Clean output
    cleaned = []
    skip_phrases = [
        "Note:", "This implementation", "Here are", "Assuming", 
        "// Generated by", "```", "/*", "*/", "<!--", "-->"
    ]
    
    for file in files:
        content = file["content"].strip()
        lines = content.splitlines()
        
        # Remove skip phrases
        cleaned_lines = [
            line for line in lines
            if not any(phrase in line for phrase in skip_phrases)
        ]
        
        # Remove code fences
        content = "\n".join(cleaned_lines)
        content = re.sub(r'```[a-z]*\n', '', content)
        
        # Special handling for feature files
        if file["path"].endswith(".feature"):
            content = "\n".join([
                line for line in content.splitlines()
                if re.match(r'^(Feature:|Scenario:|Given |When |Then |And |#)', line)
            ])
        
        cleaned.append({"path": file["path"], "content": content})
    
    return cleaned

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_script_with_llm(
    llm,
    language: str,
    tool: str,
    style: str,
    scenario: str,
    test_type: str,
    app_url: str = None,
    recorder_context: str = None,
    doc_context: str = None,
    requirements: str = None,
    existing_tests: str = None,
    timestamp: str = None  # Unique identifier for Streamlit keys
) -> list:
    """
    Generates test automation scripts with strict enforcement of required files:
    - Feature file (.feature)
    - Step definitions (.ts/.js/.py)
    - POM classes used in steps (.ts/.js/.py)
    Uses multiple validation techniques to ensure all components are present.
    """
    # --- Supported tool/language combinations ---
    SUPPORTED = {
        "playwright": {
            "typescript": {"bdd": {"step_ext": "ts", "pom_ext": "ts"}},
            "javascript": {"bdd": {"step_ext": "js", "pom_ext": "js"}},
        },
        "selenium": {
            "python": {"bdd": {"step_ext": "py", "pom_ext": "py"}},
        },
    }
    
    # Validate tool/language/style combination
    combo = (
        SUPPORTED.get(tool.lower(), {})
                 .get(language.lower(), {})
                 .get(style.lower())
    )
    if not combo:
        raise ValueError(f"Unsupported combination: {tool}/{language}/{style}")
    
    step_ext = combo["step_ext"]
    pom_ext  = combo["pom_ext"]
    
    # --- Core instructions with explicit output format ---
    output_format = textwrap.dedent(f"""
        === OUTPUT REQUIREMENTS ===
        You MUST generate output in EXACTLY this format:
        
        // BEGIN FILE: features/{test_type}.feature
        Feature: [Feature name]
          Scenario: [Scenario name]
            Given [step]
            When [step]
            Then [step]
        
        // BEGIN FILE: steps/{test_type}.{step_ext}
        import {{ Given, When, Then }} from '@cucumber/cucumber';
        import HomePage from '../../supports/pages/HomePage';  // REQUIRED: Import POM classes
        
        Given('[step pattern]', async () => {{
            await new HomePage(page).navigate();  // REQUIRED: Use POM classes
        }});
        
        // BEGIN FILE: supports/pages/HomePage.{pom_ext}
        import {{ Page }} from '@playwright/test';
        
        export default class HomePage {{
            constructor(private page: Page) {{}}
            
            async navigate() {{
                await this.page.goto('/');
            }}
        }}
        
        // END OUTPUT
    """).strip()

    # --- Build concise prompt ---
    prompt = textwrap.dedent(f"""
        ROLE: Senior Test Automation Engineer
        TASK: Generate test automation files for this scenario
        TOOL: {tool} | LANGUAGE: {language} | TEST TYPE: {test_type}
        
        === SELECTOR MAPPING (from Recorder) ===
        {recorder_context}

        === LOCATOR DERIVATION RULES ===
        1• Derive every `page.locator(…)` selector directly from the Recorder JSON above.
        2• Do NOT invent or guess any selectors—use only those present in the JSON.
        3• If a step needs a selector not in the JSON, leave a clear placeholder comment.

        === KEY RULES ===
        1. MUST generate 3 files: feature, steps, and POM classes
        2. Steps MUST import and use POM classes
        3. POM classes MUST be in supports/pages/
        4. Steps MUST include an implementation body, never leave a Given/When/Then empty
        5. Follow the EXACT output format below
        
        === SCENARIO ===
        {scenario}
        
        {output_format}
        
        IMPORTANT: 
        - DO NOT include any explanations or markdown
        - DO NOT repeat these instructions
        - START output with '// BEGIN FILE:'
        - INCLUDE ALL REQUIRED FILES
    """)
    
    logger.info(f"Prompt for {test_type}:\n{prompt[:1500]}...")
    
    # --- Invoke LLM ---
    try:
        if hasattr(llm, "invoke"):
            raw = llm.invoke(prompt, timeout=OLLAMA_TIMEOUT)
        else:
            raw = asyncio.run(llm.ainvoke(prompt, timeout=OLLAMA_TIMEOUT))

        st.write(raw)
        raw_text = getattr(raw, "content", str(raw))
        logger.info(f"Raw LLM output ({len(raw_text)} chars)")
        
        # --- Debug: Show raw output in UI ---
        st.subheader("Raw LLM Output")
        # Create unique key using timestamp
        unique_key = f"raw_{test_type}_{timestamp or time.time()}"
        st.text_area("", value=raw_text, height=400, key=unique_key)
        
        # --- Parse using explicit markers ---
        files = []
        current_path = ""
        current_content = []
        
        # Split by file markers
        sections = re.split(r'// BEGIN FILE:\s*', raw_text)
        for section in sections:
            if not section.strip():
                continue
                
            # Extract path and content
            path_end = section.find('\n')
            if path_end == -1:
                path = section.strip()
                content = ""
            else:
                path = section[:path_end].strip()
                content = section[path_end:].strip()
            
            # Remove end marker if present
            if "// END OUTPUT" in content:
                content = content.split("// END OUTPUT")[0].strip()
                
            # Validate and store
            if path:
                files.append({
                    "path": path,
                    "content": content
                })
        
        # --- Ensure required files exist ---
        required_files = {
            f"features/{test_type}.feature": {
                "exists": False,
                "content": f"# MISSING FEATURE FILE\nFeature: Placeholder\n  Scenario: Implement scenario\n    Given Implement steps\n"
            },
            f"steps/{test_type}.{step_ext}": {
                "exists": False,
                "content": f"// MISSING STEP FILE\nimport {{ Given }} from '@cucumber/cucumber';\n\nGiven('Implement steps', () => {{}});"
            }
        }
        
        pom_files = []
        
        # Check what we have
        for file in files:
            # Check for required files
            if file["path"] in required_files:
                required_files[file["path"]]["exists"] = True
            
            # Collect POM files
            if file["path"].startswith("supports/pages/") and file["path"].endswith(f".{pom_ext}"):
                pom_files.append(file)
        
        # Add missing required files
        for path, info in required_files.items():
            if not info["exists"]:
                files.append({
                    "path": path,
                    "content": info["content"]
                })
        
        # --- Critical: Ensure POM classes exist and are used in steps ---
        step_file_content = next(
            (f["content"] for f in files if f["path"] == f"steps/{test_type}.{step_ext}"), 
            ""
        )
        
        # 1. Find POM classes referenced in step definitions
        pom_imports = re.findall(
            r'import\s+\{?\s*(\w+)\s*\}?\s+from\s+[\'"]\.\./\.\./supports/pages/',
            step_file_content
        )
        logger.info(f"Detected POM imports in steps: {pom_imports}")
        
        # 2. Ensure each imported POM exists
        for pom_class in pom_imports:
            pom_path = f"supports/pages/{pom_class}.{pom_ext}"
            
            # Check if we already have this POM
            if not any(f["path"] == pom_path for f in files):
                # Create missing POM
                files.append({
                    "path": pom_path,
                    "content": (
                        f"// MISSING POM CLASS: {pom_class}\n"
                        f"import {{ Page }} from '@playwright/test';\n\n"
                        f"export default class {pom_class} {{\n"
                        f"  constructor(private page: Page) {{}}\n\n"
                        f"  // IMPLEMENT METHODS USED IN STEPS\n"
                        f"}}"
                    )
                })
                logger.warning(f"Added missing POM: {pom_path}")
        
        # 3. If no POMs at all, create default
        if not pom_imports and not pom_files:
            default_pom = "DefaultPage"
            default_path = f"supports/pages/{default_pom}.{pom_ext}"
            files.append({
                "path": default_path,
                "content": (
                    f"import {{ Page }} from '@playwright/test';\n\n"
                    f"export default class {default_pom} {{\n"
                    f"  constructor(private page: Page) {{}}\n\n"
                    f"  async navigate() {{\n"
                    f"    await this.page.goto('/');\n"
                    f"  }}\n"
                    f"}}"
                )
            })
            
            # Update step file to import default POM
            step_path = f"steps/{test_type}.{step_ext}"
            for file in files:
                if file["path"] == step_path:
                    file["content"] = (
                        f"import {{ Given }} from '@cucumber/cucumber';\n"
                        f"import {default_pom} from '../../supports/pages/{default_pom}';\n\n"
                        f"{file['content']}"
                    )
        
        # --- Post-process content ---
        for file in files:
            content = file["content"]
            
            # Remove comment-only lines
            content = re.sub(r'^\s*//.*$', '', content, flags=re.MULTILINE)

            # Remove REQUIRED comments
            content = re.sub(r'//\s*REQUIRED:.*$', '', content, flags=re.MULTILINE)
            
            # Remove markdown code fences
            content = re.sub(r'```[a-z]*', '', content)
            
            # Special handling for feature files
            if file["path"].endswith(".feature"):
                # Keep only Gherkin lines
                content = "\n".join([
                    line for line in content.splitlines() 
                    if re.match(r'^\s*(Feature|Scenario|Given|When|Then|And|#)', line, re.IGNORECASE)
                ])
            
            file["content"] = content.strip()
        
        # --- Log results ---
        logger.info(f"Generated {len(files)} files:")
        for file in files:
            logger.info(f" - {file['path']} ({len(file['content'])} chars)")
        
        return files
        
    except Exception as err:
        logger.error(f"LLM processing failed: {err}")
        st.error(f"⚠️ LLM processing error: {str(err)}")
        return [
            {"path": f"features/{test_type}.feature", "content": "# ERROR: Generation failed"},
            {"path": f"steps/{test_type}.{step_ext}", "content": "// ERROR: Generation failed"},
            {"path": f"supports/pages/ErrorPage.{pom_ext}", "content": "// ERROR: Generation failed"}
        ]

def extract_selectors_from_recorder(recorder_json: dict) -> dict:
    """
    Given a DevTools Recorder JSON object, extract a mapping from
    “command + value” to the actual selector used.

    Example input step:
      { "command": "click", "target": "#searchInput", "value": "" }
    Output mapping key: "Click" or "Type dyson fan" → "#searchInput"
    """
    mapping = {}
    for step in recorder_json.get("steps", []):
        cmd = step.get("command", "").capitalize()
        val = step.get("value", "")
        # Build a descriptive key: e.g. "Type dyson fan" or just "Click"
        key = f"{cmd} {val}".strip()
        selector = step.get("target") or step.get("selector") or ""
        if key and selector:
            mapping[key] = selector
    return mapping

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
selected_tool = st.sidebar.selectbox("Tool", ["Playwright", "Selenium"])
selected_lang = st.sidebar.selectbox(
    "Language", ["Python", "JavaScript", "TypeScript", "Java", "C#"]
)
selected_style = st.sidebar.radio("Style", ["Plain", "BDD"], horizontal=True)

# Define output_dir *before* we use it below
target_path = st.sidebar.text_input(
    "Output Path", value=str(Path.cwd() / "auto_project")
)
output_dir = Path(target_path)

# Base URL selection
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
# 🔧 Generate Automation Scripts from Stored Test Cases
if st.sidebar.button("Generate Automation Scripts (Stored Only)"):
    with st.spinner("🚀 Generating automation scripts..."):
        allowed_types = ["positive", "negative", "edge", "accessibility"]

        # 1) Scaffold framework if needed
        if not detect_existing_framework(output_dir, selected_lang, selected_tool, selected_style):
            scaffold_framework(output_dir, selected_lang, selected_tool, selected_style)
            logger.info("Scaffolded new automation framework.")
            st.sidebar.info("Scaffolded new automation framework.")

        # 2) Load contexts
        llm = get_llm(st.session_state["model_selector"])
        doc_ctx, reqs, existing_tests, raw_recorder_ctx = get_hybrid_context(
            st.session_state.vector_store, with_recorder=True
        )
        logger.info(f"Loaded existing test steps:\n{existing_tests}")

        # 3) Build selector snippet from recorder JSON
        recorder_ctx = ""
        if raw_recorder_ctx:
            try:
                recorder_json = json.loads(raw_recorder_ctx)
                selector_map = extract_selectors_from_recorder(recorder_json)
                recorder_ctx = "\n".join(f"{k} → {v}" for k, v in selector_map.items())
            except Exception as e:
                logger.warning(f"Failed to parse recorder JSON: {e}")
                recorder_ctx = raw_recorder_ctx

        # 4) Group test cases by explicit tag
        results = st.session_state.vector_store.get()
        grouped = {t: [] for t in allowed_types}
        for m, doc in zip(results["metadatas"], results["documents"]):
            if m.get("source_type") == "test_case" and m.get("test_type") in allowed_types:
                grouped[m["test_type"]].append(doc)
        logger.info("Grouped scenarios: " +
                    ", ".join(f"{t}={len(v)}" for t, v in grouped.items()))

        # 5) Only support Playwright+TS+BDD here
        if (
            selected_tool.lower() == "playwright"
            and selected_lang.lower() == "typescript"
            and selected_style.lower() == "bdd"
        ):
            # ensure directories exist
            (output_dir / "features").mkdir(parents=True, exist_ok=True)
            (output_dir / "steps").mkdir(parents=True, exist_ok=True)
            (output_dir / "supports" / "pages").mkdir(parents=True, exist_ok=True)

            # 6) Per‐category, per‐scenario generation
            for test_type in allowed_types:
                scenarios = grouped[test_type]
                if not scenarios:
                    st.sidebar.info(f"No `{test_type}` scenarios.")
                    continue

                # prepare feature + step file paths
                feat_file = output_dir / "features" / f"{test_type}.feature"
                step_file = output_dir / "steps"    / f"{test_type}.ts"
                pom_dir   = output_dir / "supports" / "pages"

                # Initialize step content
                step_content = step_file.read_text(encoding="utf-8") if step_file.exists() else ""
                
                # scaffold files if missing
                if not feat_file.exists():
                    feat_file.write_text(f"Feature: {test_type.capitalize()} tests\n\n")
                if not step_file.exists():
                    step_file.write_text("")

                st.write(f"## `{test_type}` scenarios")
                for idx, scenario in enumerate(scenarios, start=1):
                    st.write(f"### Scenario {idx}: {scenario.splitlines()[0]}")

                    # a) append Gherkin
                    feat_file.write_text(
                        feat_file.read_text(encoding="utf-8") + scenario + "\n\n"
                    )

                    # b) invoke LLM for this one scenario
                    with st.spinner(f"Generating code for scenario {idx}…"):
                        # Generate unique timestamp for this scenario
                        unique_ts = str(time.time())
                        files = generate_test_script_with_llm(
                            llm=llm,
                            language=selected_lang,
                            tool=selected_tool,
                            style=selected_style,
                            scenario=scenario,
                            test_type=test_type,
                            app_url=selected_app_url,
                            recorder_context=recorder_ctx,
                            doc_context=doc_ctx,
                            requirements=reqs,
                            existing_tests=existing_tests,
                            timestamp=unique_ts  # Add unique timestamp
                        )

                    # c) process each returned file
                    for f in files:
                        path = output_dir / f["path"]
                        path.parent.mkdir(parents=True, exist_ok=True)

                        # Feature file - already appended above
                        if f["path"].startswith("features/"):
                            continue

                        # Step definitions - merge with existing content
                        elif f["path"].startswith("steps/"):
                            # Get generated step content
                            generated_content = f["content"].strip()
                            st.write(generated_content)
                            # Merge with existing step content
                            step_content += "\n\n" + generated_content
                            continue

                        # POM classes - create only if missing
                        elif f["path"].startswith("supports/pages/"):
                            if not path.exists():
                                path.write_text(f["content"].strip(), encoding="utf-8")
                            continue

                    st.success(f"✅ Scenario {idx} done.")

                # Remove duplicate imports while preserving step definitions
                step_content = remove_duplicate_imports(step_content)
                
                # Write consolidated step definitions
                step_file.write_text(step_content, encoding="utf-8")
                st.sidebar.success(f"`{test_type}` scripts updated.")
                
                # Debug: Show step content
                st.subheader(f"Step Definitions for {test_type}")
                st.code(step_content)
        else:
            st.sidebar.error("Only Playwright + TypeScript + BDD is supported for automation scripts.")

        st.sidebar.success(f"All scripts generated in {output_dir}")











