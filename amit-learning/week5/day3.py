from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr 
import os, glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL= "gpt-6-luna"
DB_NAME="vector_db"
KB_PATH=Path(__file__).resolve().parent / "knowledge-base"
load_dotenv()

folders= KB_PATH.glob("*")

documents=[]
for folder in folders:
    doc_type=os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs= loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks= text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(DB_NAME):
    Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

vector_store= Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_NAME
)

retriever = vector_store.as_retriever()
llm= ChatOpenAI(temperature=1, model_name=MODEL)

print(retriever.invoke('Who is Avery?'))

# print(llm.invoke("Who is Avery?"))


SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

def answer_question(question: str, history):
    docs= retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response=llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content

# print(answer_question("Who is Averi Lancaster?", []))

gr.ChatInterface(answer_question).launch(inbrowser=True)