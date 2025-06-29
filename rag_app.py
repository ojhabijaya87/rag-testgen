# rag_app.py

import streamlit as st
import asyncio
import time
import os
import datetime
import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import SeleniumURLLoader

from rag_models import MODEL_CONFIG, MODEL_USAGE_HINTS
from rag_prompts import TEST_PROMPT, TEST_TYPE_INSTRUCTIONS, filter_test_context
from rag_context import get_hybrid_context, get_existing_test_cases
from rag_helpers import (
    anonymize_story,
    get_story_hash,
    split_scenarios,
    create_scenario_documents
)

try:
    from langchain_community.llms.ollama import Ollama
except ImportError:
    from langchain_community.llms import Ollama

from langchain_openai import ChatOpenAI

# --- EMBEDDINGS ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# --- LLM Selection ---
def get_llm(model_name):
    if model_name == "Select a model":
        return None

    config = MODEL_CONFIG[model_name]
    provider = config["provider"]

    if provider == "ollama":
        return Ollama(
            model=config["model_name"],
            temperature=config["temperature"],
            num_predict=config["max_tokens"],
            top_p=config["top_p"]
        )
    elif provider == "openai":
        return ChatOpenAI(
            model_name=config["model_name"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            model_kwargs={"top_p": config["top_p"]}
        )
    else:
        raise ValueError(f"Provider {provider} not implemented yet.")

# --- RAG Generation ---
async def generate_prompt_async(prompt, llm):
    return await llm.ainvoke(prompt)

async def generate_test_type(test_type, context, requirements, existing_tests, current_story, llm):
    filtered_tests = filter_test_context(existing_tests, test_type)
    full_prompt = TEST_PROMPT.format(
        context=context[:300],
        requirements=requirements[:200],
        existing_tests=filtered_tests[:400],
        current_story=current_story,
        test_type_instructions=TEST_TYPE_INSTRUCTIONS[test_type]
    )
    return await generate_prompt_async(full_prompt, llm)

async def generate_all_tests(context, requirements, existing_tests, current_story, llm):
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

# --- Initialize Vector Store ---
if "vector_store" not in st.session_state:
    if os.path.exists("./chroma_db"):
        st.session_state.vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        st.session_state.vector_store = None

# --- UI CONFIG ---
st.set_page_config(
    page_title="AI Test Case Generator",
    page_icon="🧪",
    layout="wide"
)

# --- HEADER ---
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4CAF50;'>🧪 AI Test Case Generator</h1>
        <p style='font-size:18px; color: #555;'>
            Generate, edit, and manage BDD-style test cases from your user stories and documentation.
        </p>
    </div>
""", unsafe_allow_html=True)

st.write("")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model_names = list(MODEL_CONFIG.keys())
    model_name = st.selectbox("Select Model", model_names, index=0)
    st.caption(MODEL_USAGE_HINTS.get(model_name, ""))

    st.divider()

    st.subheader("🔗 Import Documentation")
    url_input = st.text_area("Enter one or more URLs:")

    if st.button("Process URLs"):
        docs = []
        for url in url_input.splitlines():
            url = url.strip()
            if not url:
                continue
            loader = SeleniumURLLoader(urls=[url])
            loaded_docs = loader.load()
            timestamp = datetime.datetime.now().isoformat()
            for doc in loaded_docs:
                doc.metadata.update({
                    "source_url": url,
                    "ingested_at": timestamp,
                    "source_type": "documentation"
                })
            docs.extend(loaded_docs)

        if docs:
            if st.session_state.vector_store:
                st.session_state.vector_store.add_documents(docs)
            else:
                st.session_state.vector_store = Chroma.from_documents(
                    docs,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
            st.success(f"✅ Loaded {len(docs)} documents.")

# --- USER STORY INPUT ---
st.subheader("📝 Enter Your User Story")

with st.form("user_story_form"):
    user_story = st.text_area(
        "Paste your user story below:",
        height=150,
        placeholder="e.g. As a user, I want to edit journeys so I can change my departure time."
    )
    submitted = st.form_submit_button("🚀 Generate Test Cases")

    if submitted:
        st.session_state.generating = True
        start_time = time.perf_counter()

        llm = get_llm(model_name)
        anonymized = anonymize_story(user_story)
        story_hash = get_story_hash(anonymized)

        existing_tests = get_existing_test_cases(
            st.session_state.vector_store,
            story_hash
        ) if st.session_state.vector_store else None

        if existing_tests and any(bool(v.strip()) for v in existing_tests.values()):
            st.session_state.edited_tests = existing_tests
            st.toast("✅ Loaded existing test cases.", icon="💾")
        else:
            doc_context, requirements, existing_test_steps = get_hybrid_context(
                st.session_state.vector_store
            ) if st.session_state.vector_store else ("", "", "")

            st.session_state.rag_context = {
                "documentation": doc_context,
                "requirements": requirements,
                "existing_tests": existing_test_steps
            }

            results = asyncio.run(
                generate_all_tests(doc_context, requirements, existing_test_steps, user_story, llm)
            )

            positive, negative, edge, accessibility = results

            st.session_state.raw_tests = {
                "positive": positive,
                "negative": negative,
                "edge": edge,
                "accessibility": accessibility
            }
            st.session_state.edited_tests = st.session_state.raw_tests

        elapsed = time.perf_counter() - start_time
        st.info(f"⏱️ Completed in {elapsed:.1f} seconds.")
        st.session_state.generating = False

# --- DISPLAY TEST CASES ---
if "edited_tests" in st.session_state and st.session_state.edited_tests:
    st.subheader("✏️ Edit & Save Test Cases")

    tabs = st.tabs(["✅ Positive", "❌ Negative", "🟧 Edge", "♿ Accessibility"])

    for test_type, tab in zip(["positive", "negative", "edge", "accessibility"], tabs):
        with tab:
            color_map = {
                "positive": "#DFF2BF",
                "negative": "#FFBABA",
                "edge": "#FEEFB3",
                "accessibility": "#BDE5F8"
            }
            st.markdown(
                f"<div style='background-color:{color_map[test_type]}; padding:10px; border-radius:5px;'>"
                f"<strong>{test_type.capitalize()} Test Cases</strong>"
                f"</div>",
                unsafe_allow_html=True
            )
            edited_text = st.text_area(
                "",
                value=st.session_state.edited_tests.get(test_type, ""),
                height=300,
                key=f"edit_{test_type}"
            )
            st.session_state.edited_tests[test_type] = edited_text

            if st.button(f"💾 Save {test_type.capitalize()} Tests", key=f"save_{test_type}"):
                if st.session_state.vector_store:
                    anonymized = anonymize_story(user_story)
                    story_hash = get_story_hash(anonymized)

                    scenarios = split_scenarios(edited_text)
                    if not scenarios:
                        st.warning("No scenarios found to save.")
                    else:
                        docs = create_scenario_documents(scenarios, test_type, story_hash)

                        for doc in docs:
                            new_hash = doc.metadata["content_hash"]

                            # Check if identical doc exists
                            existing = st.session_state.vector_store.get(
                                where={"content_hash": {"$eq": new_hash}}
                            )
                            if existing["documents"]:
                                st.info(f"✅ Scenario already stored: {doc.page_content[:50]}...")
                                continue

                            # Check if any older doc exists for same story and test type
                            # Check if any older doc exists for same story and test type
                        old_docs_all = st.session_state.vector_store.get(
                            where={"related_story_hash": {"$eq": story_hash}}
                            )

                        old_docs = {
                            "documents": [],
                            "metadatas": []
                            }
                        for doc, meta in zip(old_docs_all["documents"], old_docs_all["metadatas"]):
                            if meta.get("test_type") == test_type:
                                old_docs["documents"].append(doc)
                                old_docs["metadatas"].append(meta)


                            duplicate_found = False
                            for old_doc, old_meta in zip(old_docs["documents"], old_docs["metadatas"]):
                                if old_doc.strip() != doc.page_content.strip():
                                    old_hash = old_meta.get("content_hash")
                                    if old_hash:
                                        st.session_state.vector_store.delete(
                                            where={"content_hash": old_hash}
                                        )
                                        st.info(f"🗑️ Deleted old scenario: {old_doc[:50]}...")
                                    duplicate_found = True

                            # Save the new scenario
                            st.session_state.vector_store.add_documents([doc])
                            if duplicate_found:
                                st.success(f"✅ Updated scenario: {doc.page_content[:50]}...")
                            else:
                                st.success(f"✅ Saved new scenario: {doc.page_content[:50]}...")
                else:
                    st.warning("Vector store not initialized.")

# --- RAG CONTEXT ---
if "rag_context" in st.session_state:
    st.subheader("🧠 RAG Context Used for Generation")

    for k in ["documentation", "requirements", "existing_tests"]:
        st.subheader(f"{k.capitalize()} Context")
        ctx = st.session_state.rag_context.get(k, "")
        if ctx:
            st.info(ctx)
        else:
            st.warning("No context available.")

# --- DATA DASHBOARD ---
if st.session_state.vector_store:
    st.subheader("📊 Stored Data Dashboard")

    collection = st.session_state.vector_store.get()
    metadatas = collection.get("metadatas", [])
    documents = collection.get("documents", [])

    # --- User Stories ---
    user_stories = [
        doc for doc, meta in zip(documents, metadatas)
        if meta.get("source_type") == "user_story"
    ]
    with st.expander(f"📜 User Stories ({len(user_stories)})", expanded=False):
        if user_stories:
            for story in user_stories:
                st.markdown(f"- {story[:200]}{'...' if len(story) > 200 else ''}")
        else:
            st.info("No user stories stored yet.")

    # --- Test Cases (GROUPED) ---
    test_cases_by_type = {
        "positive": [],
        "negative": [],
        "edge": [],
        "accessibility": []
    }

    for doc, meta in zip(documents, metadatas):
        if meta.get("source_type") == "test_case":
            ttype = meta.get("test_type", "")
            if ttype in test_cases_by_type:
                test_cases_by_type[ttype].append(doc)

    total_test_cases = sum(len(v) for v in test_cases_by_type.values())

    with st.expander(f"🧪 Test Cases ({total_test_cases})", expanded=False):
        if total_test_cases:
            for ttype in ["positive", "negative", "edge", "accessibility"]:
                cases = test_cases_by_type[ttype]
                if cases:
                    st.markdown(f"### {ttype.capitalize()} Tests ({len(cases)})")
                    for test in cases:
                        st.code(test, language="gherkin")
                        st.write("---")
        else:
            st.info("No test cases stored yet.")

    # --- Documents ---
    docs = [
        (meta.get("source_url", "N/A"), doc)
        for doc, meta in zip(documents, metadatas)
        if meta.get("source_type") == "documentation"
    ]
    with st.expander(f"📚 Documents ({len(docs)})", expanded=False):
        if docs:
            for url, content in docs:
                st.markdown(f"**URL:** {url}")
                st.write(content[:500] + ("..." if len(content) > 500 else ""))
                st.write("---")
        else:
            st.info("No documentation stored yet.")
