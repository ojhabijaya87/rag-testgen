# rag_prompts.py

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
        "Generate only positive test scenarios. Valid inputs → successful outcomes. "
        "No negative, edge, or accessibility cases. "
        "Each scenario must reflect a successful user journey."
    ),
    "negative": (
        "Generate only negative test scenarios. Invalid inputs → specific failures. "
        "Do not include positive, edge, or accessibility tests."
    ),
    "edge": (
        "Generate only edge-case test scenarios. These should cover boundary conditions, extreme values, or unusual usage. "
        "No positive, negative, or accessibility tests."
    ),
    "accessibility": (
        "Generate only accessibility test scenarios, covering WCAG 2.1 AA criteria. "
        "Do not include functional tests."
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
