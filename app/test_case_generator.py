from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from app.prompt_templates import test_case_prompt  # <-- Adjust path as needed

def generate_test_cases(requirement_text, retriever):
    llm = Ollama(model=os.getenv("LLM_MODEL", "mistral:7b"))
    prompt = PromptTemplate.from_template(test_case_prompt)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.run(requirement_text)

# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM


# def generate_test_cases(query, retriever):
#     # Use invoke instead of get_relevant_documents
#     relevant_docs = retriever.invoke(query)
#     context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
#     # Use OllamaLLM instead of Ollama
#     prompt = PromptTemplate.from_template(test_case_prompt)
#     llm = OllamaLLM(model="mistral:7b")
#     chain = prompt | llm
#     result = chain.invoke({"context": context_text})
#     return result
