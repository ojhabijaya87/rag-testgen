# rag_helpers.py

import re
import hashlib
import datetime
from langchain_core.documents import Document

def anonymize_story(story: str) -> str:
    story = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+', 'User', story)
    story = re.sub(r'\S+@\S+', 'user@example.com', story)
    story = re.sub(r'\b\d{10}\b', 'XXXXXXXXXX', story)
    return story

def get_story_hash(story: str) -> str:
    return hashlib.sha256(story.encode()).hexdigest()

def split_scenarios(text):
    """
    Splits text into individual BDD scenarios.
    """
    if not text.strip():
        return []
    splits = re.split(r'(?=^Scenario:)', text, flags=re.MULTILINE)
    return [s.strip() for s in splits if s.strip()]

def create_scenario_documents(scenarios, test_type, story_hash):
    """
    Creates Document objects for each scenario, adding a unique content hash.
    """
    docs = []
    for scenario in scenarios:
        content_hash = hashlib.sha256(scenario.encode()).hexdigest()
        docs.append(
            Document(
                page_content=scenario,
                metadata={
                    "source_type": "test_case",
                    "test_type": test_type,
                    "related_story_hash": story_hash,
                    "content_hash": content_hash,
                    "created_at": datetime.datetime.now().isoformat(),
                }
            )
        )
    return docs
