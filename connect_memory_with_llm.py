import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv ## Uncomment the following files if you're not using pipenv as your virtual environment manager


load_dotenv(find_dotenv())












#step 1: Setup LLM (Mistral with HuggingFace)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

def load_llm(model_name: str)-> ChatOpenAI:
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        max_tokens=512,
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",          
        )
    return llm

#step 2: Connect LLM with FAISS and Create Chain 
DB_FAISS_PATH="vector_store/db_faiss"



custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the context.
Content:{context}
Question: {question}

Start the ansewer directly. No small talk please.
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt 


#load Database
DB_FAISS_PATH="vector_store/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)


#create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(OPENROUTER_MODEL_NAME),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k":3}),
    chain_type_kwargs={"prompt": set_custom_prompt()},
    return_source_documents=True
)


# Now invoke the chain with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print("RESULT: ", response['result'])
print("SOURCE DOCUMENTS: ", response['source_documents'])