# rag_context.py

import streamlit as st

def get_hybrid_context(vector_store):
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


def get_existing_test_cases(vector_store, story_hash):
    if not vector_store:
        return None

    test_cases = {t: "" for t in ["positive", "negative", "edge", "accessibility"]}

    try:
        results = vector_store.get(
            where={
                "$and": [
                    {"source_type": {"$eq": "test_case"}},
                    {"related_story_hash": {"$eq": story_hash}}
                ]
            }
        )
        if 'metadatas' in results and 'documents' in results:
            for doc, meta in zip(results['documents'], results['metadatas']):
                test_type = meta.get('test_type', '')
                if test_type in test_cases and doc:
                    test_cases[test_type] += doc.strip() + "\n\n"

        # Clean up whitespace
        for t in test_cases:
            test_cases[t] = test_cases[t].strip()

        return test_cases
    except Exception as e:
        st.error(f"Error loading existing test cases: {str(e)}")
        return None
