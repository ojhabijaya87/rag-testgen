# TfL Journey Planner Test Generator (Final Version)
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

# Timeout settings
OLLAMA_TIMEOUT = 400

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

# Enhanced BDD Prompt Templates with Gherkin Features
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
    
    # Prepare all three prompts
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
    
    return results  # Returns [positive, negative, edge]

def get_hybrid_context(vector_store, feature_name):
    """Retrieve both TfL docs and user stories"""
    tfl_context = ""
    user_requirements = ""
    
    if not vector_store:
        return tfl_context, user_requirements
    
    try:
        # Get official TfL context
        tfl_docs = vector_store.similarity_search(
            feature_name, 
            k=2,
            filter=lambda meta: meta.get("source_type") == "tfl"
        )
        tfl_context = "\n\n".join(doc.page_content[:300] for doc in tfl_docs)
        
        # Get relevant user stories
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
    """Remove sensitive information from user stories"""
    # Anonymize names
    story = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+', 'User', story)
    # Remove emails
    story = re.sub(r'\S+@\S+', 'user@example.com', story)
    # Remove phone numbers
    story = re.sub(r'\b\d{10}\b', 'XXXXXXXXXX', story)
    return story

# Streamlit UI
st.set_page_config(page_title="TfL Requirements-Driven Test Generator", layout="wide")
st.title("🚇 TfL Journey Planner - Requirements-Driven Test Generator")

embeddings = load_embeddings()

with st.sidebar:
    st.header("⚙️ Configuration")
    model_names = list(MODEL_CONFIG.keys())
    default_index = model_names.index("zephyr-7b-beta (Hugging Face Hosted)") if "zephyr-7b-beta (Hugging Face Hosted)" in model_names else 0
    model_name = st.selectbox("Select Model", model_names, index=default_index, key="model_selector")
    
    st.subheader("🔗 TfL Website URL")
    url_input = st.text_area("Enter one or more TfL URLs:",
                             value="https://tfl.gov.uk/plan-a-journey/",
                             key="url_input")
    
    # User story storage toggle
    store_stories = st.checkbox("📚 Store user stories for context", value=True,
                              help="Improves test relevance by remembering past requirements")
    
    # Display stored requirements count
    if "vector_store" in st.session_state:
        try:
            stories = [doc for doc in st.session_state.vector_store.docstore._dict.values()
                      if doc.metadata.get("source_type") == "user_story"]
            st.metric("Stored Requirements", f"{len(stories)} user stories")
        except:
            pass
    
    if st.button("Process URL(s)", key="process_button"):
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
    
    # Quality indicator
    if user_story:
        story_quality = min(100, len(user_story) // 2)  # Simple quality heuristic
        st.progress(story_quality, text=f"Requirement detail: {story_quality}%")
    
    submitted = st.form_submit_button("🚀 Generate BDD Test Cases")
    
    if submitted:
        if model_name == "Select a model":
            st.warning("⚠️ Please select a model to proceed.")
        elif not user_story.strip():
            st.warning("⚠️ Please enter a user story.")
        else:
            # Start timing
            start_time = time.perf_counter()
            
            with st.spinner("🧠 Generating requirements-driven test cases..."):
                llm = get_llm(model_name)
                feature_name = "Journey Planning"
                
                # Store user story in vector DB if enabled
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
                        else:
                            st.session_state.vector_store.add_documents([story_doc])
                            
                        st.toast("📝 Stored user story for future context", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to store user story: {str(e)}")
                
                # Retrieve hybrid context
                tfl_context, user_requirements = "", ""
                if "vector_store" in st.session_state and st.session_state.vector_store:
                    tfl_context, user_requirements = get_hybrid_context(
                        st.session_state.vector_store, 
                        feature_name
                    )
                
                # Create new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Run async generation
                    positive_cases, negative_cases, edge_cases = loop.run_until_complete(
                        generate_all_tests(tfl_context, user_requirements, user_story, llm)
                    )
                    
                    # Calculate generation time
                    end_time = time.perf_counter()
                    generation_time = end_time - start_time
                    mins, secs = divmod(generation_time, 60)
                    time_str = f"{int(mins)}m {secs:.2f}s"
                    
                    # Format results
                    final_result = (
                        "### ✅ Positive Scenarios\n\n" + 
                        "```gherkin\n" + positive_cases + "\n```" +
                        "\n\n---\n\n### ❌ Negative Scenarios\n\n" + 
                        "```gherkin\n" + negative_cases + "\n```" +
                        "\n\n---\n\n### ⚠️ Edge Case Scenarios\n\n" + 
                        "```gherkin\n" + edge_cases + "\n```"
                    )
                    
                    st.session_state.generated_tests = final_result
                    st.session_state.generation_time = time_str
                    
                    st.success(f"✅ Generated comprehensive test cases in {time_str}!")
                except Exception as e:
                    st.error(f"🚨 Generation failed: {str(e)}")
                finally:
                    loop.close()

if "generated_tests" in st.session_state:
    st.subheader("🧪 Generated BDD Test Cases")
    
    if "generation_time" in st.session_state:
        st.caption(f"⏱️ Generation time: {st.session_state.generation_time}")
    
    st.markdown(st.session_state.generated_tests, unsafe_allow_html=True)
    
    # Add traceability report
    with st.expander("🔍 Traceability Report"):
        if "vector_store" in st.session_state:
            try:
                st.write("**Relevant requirements used:**")
                sources = [
                    doc.metadata["source_url"] 
                    for doc in st.session_state.vector_store.similarity_search(
                        user_story, k=3,
                        filter=lambda meta: meta.get("source_type") == "user_story"
                    )
                ]
                for src in set(sources):
                    st.write(f"- {src}")
            except:
                st.write("No requirement sources available")
    
    # Clipboard and regeneration
    if st.button("📋 Copy to Clipboard", key="copy_button"):
        clean_content = "\n\n".join([
            part.replace("```gherkin", "").replace("```", "").strip()
            for part in st.session_state.generated_tests.split("```")
            if "gherkin" not in part
        ])
        pyperclip.copy(clean_content)
        st.toast("📋 Copied BDD scenarios to clipboard!", icon="✅")
    
    if st.button("🔄 Regenerate", key="regenerate_button"):
        keys = ["generated_tests", "generation_time"]
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Add troubleshooting section
with st.expander("🛠️ Troubleshooting Guide"):
    st.markdown("""
    **Common Issues & Solutions:**
    
    1. **Tests not generating:**
       - Ensure Ollama is running: `ollama serve`
       - Download required models: `ollama pull phi`
       - Try Hugging Face model instead
       - Simplify your user story
       
    2. **Poor quality tests:**
       - Add more context URLs
       - Make user stories more detailed
       - Try a different model
       
    3. **Timeout errors:**
       - Use smaller models (phi instead of mistral)
       - Reduce context size
       - Use Hugging Face hosted model
       
    4. **BDD formatting issues:**
       - Check prompt instructions
       - Verify model supports Gherkin
       - Simplify complex scenarios
    """)