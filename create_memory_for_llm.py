from langchain.document_loaders import PyPDFLoader, DirectoryLoader #for loading pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter #for splitting large text into smaller chunks
from langchain_huggingface import HuggingFaceEmbeddings #for creating vector embeddings
from langchain_community.vectorstores import FAISS #for storing vector embeddings

#step:1 load raw pdf(s)
DATA_PATH ="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf", 
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


documents = load_pdf_files(data=DATA_PATH)
print(f"length of PDF pages:", len(documents))


# step:2 Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50 )
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

chunks = create_chunks(extracted_data=documents)
print(f"length of text_chunks:", len(chunks))


# step:3 Create Vector Embeddings

def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    return embedding

embedding_model=get_embedding_model()


# step:4 Store embeddings in FAISS(Facebbok AI Similarity Search)

DB_FAISS_PATH="vector_store/db_faiss"
db=FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)