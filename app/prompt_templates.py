test_case_prompt = """
**Role**: Senior QA Engineer specializing in context-aware test generation

**Input Context**:
{context}

**Output Requirements**:
1. Generate test cases covering:
   - Positive scenarios (happy paths)
   - Negative scenarios
   - Edge cases
   - End-to-end flows
   - Cross-functional integration points
2. Format using BDD (Given/When/Then)
3. Reference specific acceptance criteria IDs (e.g., AC1.3)
4. Include severity ratings (Critical/High/Medium/Low)
5. Add traceability matrix linking tests to requirements

**Test Case Structure**:

[Functional Area]
TC-[ID]: [Title]
Severity: [Level]
Linked to: [Story ID], [AC IDs]
Steps:
Given [initial state]
When [action/event]
Then [expected outcome]

Test Data:

Valid: [examples]
Invalid: [examples]

**Generation Rules**:
- Create 2 positive + 3 negative tests per user story
- Include 1 edge case per epic
- Generate 1 E2E flow spanning 3+ user stories
- Use actual data examples from acceptance criteria
- Reference UI elements from provided wireframes/mockups
- Consider non-functional aspects (performance, security)

**Example Output**:

Input Validation
TC-1.1: Empty Field Submission
Severity: High
Linked to: JNY-001-US1, AC1.3
Steps:
Given I'm on the journey planning form
When I leave "From" field empty and submit
Then display error "Please select a valid location"

Test Data:
Valid: "NW1 2DB", "London Eye"
Invalid: "", "XYZ 123"
"""
