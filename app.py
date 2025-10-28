from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
load_dotenv()

# TITLE OF THE APPLICATION
st.title("RAG APPLICATION USING LANGCHAIN AND STREAMLIT")

# URL INPUT FROM THE USER
url=st.text_input("ENTER THE URL")
loader=UnstructuredURLLoader(urls=[url])
data=loader.load()
st.write("DATA LOADED SUCCESSFULLY")

# TEXT SPLITTING
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs=text_splitter.split_documents(data)
all_split=docs
st.write("TEXT SPLIT COMPLETED")

# EMBEDDINGS
embeddings=OpenAIEmbeddings()
vectorstore=FAISS.from_documents(documents=all_split, embedding=embeddings)
st.write("EMBEDDINGS CREATED")

# RETRIEVAL
retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
st.write("RETRIEVER READY")

# INITIALIZING LLM
llm=OpenAI(temperature=0.7, max_tokens=500)
st.write("LLM INITIALIZED")

# QUERY
query=st.text_input("ENTER YOUR QUERY")
if query:
    response=retriever.invoke(query)
    answer=llm.invoke(f"Answer the question based on the context provided: {response[0].page_content} \n Question: {query}")
    st.write("Answer Generated:", answer)
    