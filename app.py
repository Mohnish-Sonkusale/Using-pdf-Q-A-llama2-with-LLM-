from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st

# Create LLM llama2 model with CTransformers
llm = CTransformers(
      model = "llama-2-7b-chat.ggmlv3.q8_0.bin", #download the model save in locally and mention here path model https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin 
      model_type = "llama",
      max_new_tokens = 512,
      temperature = 0.5
  )

# Initialize instructor embeddings using the Hugging Face model and mention vectorbd_path
DATA_PATH = r"\data"  # Choose your file path
DB_FAISS_PATH = "db_faiss_index"   # create faiss index folder locally 
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")  # we can use Instruct_transformer and sentence-transformers

def create_vector():
    # load the PDF file
    loader = DirectoryLoader(path=DATA_PATH, glob="*.pdf", loader_cls = PyPDFLoader)
    data = loader.load()

    # Creat the chunks by RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text = text_splitter.split_documents(data)

    # creat embeddings and save in local folder
    db = FAISS.from_documents(documents = text, embedding = embeddings)
    db.save_local(DB_FAISS_PATH)

def QA_retriever ():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(DB_FAISS_PATH, embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(search_kwargs = {"k":2})

    prompt_template = """Use following pieces of information to answer the user's question.
        try to provide as much text as possible from "response". If you dont know the answer, please just say 
        "I dont know the answer". Don't try to make up an answer.
    
        Context : {context},
        Question : {question}
    
    Only return correct and helpful answer below and nothing else.
    Helful answer:

"""

    prompt = PromptTemplate(template=prompt_template, input_variables= ["context", "question"])


    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                    retriever = retriever,
                                    input_key = "query",
                                    return_source_documents = True,
                                    chain_type_kwargs = {"prompt":prompt})

    return chain

if __name__ == "__main__":
    # create_vector()            # first run this function to create the vectors
    question = QA_retriever()    # Second call QA_retriever function

    print(question("what is market overview in november 2023?"))

print("app.py file load completely")