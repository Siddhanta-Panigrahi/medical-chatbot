# Set up UI for chatbot using Streamlit
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import streamlit as st
# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH="vector_store/db_faiss"
#@st.cache_resource # Cache the loaded FAISS index to avoid reloading on every interaction

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db 

# Define custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the context.
        Content: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,input_variables=["context", "question"]
    )
    return prompt 

# Setup LLM (Mistral with HuggingFace)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

def load_llm(model_name: str, openrouter_api_key: str) -> ChatOpenAI:
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        max_tokens=512,
        api_key=openrouter_api_key, 
        base_url="https://openrouter.ai/api/v1",          
        )
    return llm


   


def main():


   

    # Main title
    

    st.title(" 💬Ask Chatbot !")
    st.write("Welcome to the Chatbot! How can I assist you today?")

    # for show your chat history used session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your AI Assistant . How can I help you ?"
        }]
        
        # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
            
                
    prompt = st.chat_input("Pass your prompt here:") # get user input
    
    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})


          
    


    try:
         with st.spinner("Processing..."):
            vectorstore = get_vectorstore()
            if vectorstore is None:
             st.error("Failed to load the vector store.")
             st.stop() # Stop execution if vector store is not available
            #create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(OPENROUTER_MODEL_NAME,openrouter_api_key=openrouter_api_key),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
                return_source_documents=True
                )
             # Invoke the chain with the user's prompt
            response = qa_chain.invoke({'query': prompt})
    
            result = response['result']
            #source_documents = response['source_documents']
            result_to_show = result #+ "\n\n**Sources:** " + str(source_documents)
    
            # Here you would typically call your LLM and get a response
            #response = "Hi, I am Medibot" # replace this with actual LLM call
            st.chat_message("assistant").markdown(result_to_show) # display the bot response in chat message container
            st.session_state.messages.append({"role": "assistant", "content": result_to_show}) # save the bot response in session state
            
            
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
if __name__ == "__main__": 
    main()
