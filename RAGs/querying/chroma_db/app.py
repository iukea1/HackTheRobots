import gradio as gr
import os
import time
import arxiv
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings

# import
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
def init_or_load_chroma(path="./tmp/local_chroma", collection_name="llm_dataset_survey_papers"):
    try:
        chroma_instance = Chroma(
            persist_directory=path,
        )
        print("Existing embeddings loaded.")
    except Exception as e:
        print("No existing embeddings found, starting new Chroma instance.", e)
        chroma_instance = None

    return chroma_instance, chroma_instance if chroma_instance else None

# Attempt to load existing Chroma instance
chroma, retriever = init_or_load_chroma()

def process_papers(query):
    """
    Process papers based on the given query.

    Args:
        query (str): The query string used to search for papers.

    Returns:
        str: A message indicating that the papers have been processed and saved to the embeddings database.
    """
    dirpath = "llm_dataset_survey_papers"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_order=arxiv.SortOrder.Descending
    )
    
    for result in client.results(search):
        while True:
            try:
                result.download_pdf(dirpath=dirpath)
                print(result)
                print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                print("Error occurred:", e)
                time.sleep(5)
    
    papers = []
    loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
    try:
        papers = loader.load()
    except Exception as e:
        print(f"Error loading file: {e}")
    full_text = ''
    for paper in papers:
        full_text += paper.page_content
    
    full_text = " ".join(line for line in full_text.splitlines() if line)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])
    
    global qdrant, retriever
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
        path="./tmp/local_qdrant",
        collection_name="llm_dataset_survey_papers",
    )
    retriever = qdrant.as_retriever()
    
    return "Papers processed and saved to embeddings database."

def perform_query(question_text):
    global retriever
    if not retriever:
        return "Error: No papers processed. Please process papers first."
    
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    ollama_llm = "mistral"
    model = ChatOllama(model=ollama_llm)
    
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    
    class Question(BaseModel):
        __root__: str
    
    chain = chain.with_types(input_type=Question)
    result = chain.invoke(Question(__root__=question_text))
    return result

with gr.Blocks() as demo:
    with gr.Tab("Process Papers"):
        query_input = gr.Text(label="Search Query")
        process_button = gr.Button("Process Papers")
        process_output = gr.Text(label="Process Output")
        process_button.click(process_papers, inputs=query_input, outputs=process_output)

    with gr.Tab("Query Retriever"):
        question_input = gr.Text(label="Question")
        query_button = gr.Button("Query")
        query_output = gr.Text(label="Query Output")
        query_button.click(perform_query, inputs=question_input, outputs=query_output)

demo.launch()