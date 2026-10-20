import os, glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
MODEL = 'gpt-6-luna'

DB_NAME= "vector_db"
KB_PATH= Path(__file__).resolve().parent.parent / "knowledge-base"

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def fetch_documents():
    folders= KB_PATH.glob("*")
    documents=[]
    for folder in folders:
        doc_type= os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"]=doc_type
            documents.append(doc)
    return documents

def create_chunks(documents):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )

    collection = vector_store._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)

    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vector_store

if __name__ == "__main__":
    documents = fetch_documents()
    chunks=create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion Complete!")