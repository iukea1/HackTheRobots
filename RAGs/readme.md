# RAGs Notebooks

This directory contains a collection of Jupyter Notebooks showcasing various aspects of Retrieval-Augmented Generation (RAG) using different libraries and techniques. 

## Notebooks Difficulty Guide

This guide provides an overview of the difficulty levels of different notebooks in the RAGs project.

**Easiest:** 

* **simple_rag.ipynb:** Provides a simple implementation of the RAGs model. Excellent starting point for beginners.

**Intermediate:** 

* **advanced_rag.ipynb:** Builds upon concepts from `simple_rag.ipynb` and introduces advanced techniques for improving RAGs model performance.

**Hardest:**

* **multi-document_agent.ipynb:** Works with multiple documents and requires a deeper understanding of the RAGs model. Recommended for experienced users familiar with the RAGs framework.

**Note:** Difficulty levels are subjective and may vary based on individual experience and familiarity with the RAGs project.

## Project Structure and Notebook Descriptions

### 1. Conversational RAG (conversational_rag.ipynb)

This notebook focuses on building a conversational RAG system. 

**Key Features:**
  * Demonstrates how to maintain conversation context using a `Memory` object.
  * Illustrates asking follow-up questions that rely on the history of the conversation.

### 2. Querying 

This subdirectory contains notebooks that demonstrate how to use RAG models to enhance querying across multiple documents.

**2.1 Multi-Document Agents (querying/Multi_Document_Agents.ipynb)**

  * Demonstrates the `MultiDocumentAgent` class for handling queries spanning multiple documents.
  * Shows how to initialize the agent, add documents, and use it for answering queries.

**2.2 SubQuestion Query Engine (querying/sub-questions-meta/SubQuestion_Query_Engine.ipynb)**
  * Tackles complex queries by breaking them down into simpler sub-queries.
  * Utilizes the `SubQuestionQueryEngine` to generate answers by combining results from sub-queries.
  * Provides a detailed walkthrough of setting up the query engine, including callback managers and debug handlers.

**2.3 App (querying/app.py)**
  * Provides a Gradio interface for querying academic papers.
  * Uses the ArXiv API to download papers, processes them, and stores them in an embeddings database.
  * Allows users to perform semantic search on the processed papers.

**2.4 Semantic Search (querying/semantic-search.ipynb)**
* Demonstrates how to use Pinecone for semantic search.
* Showcases downloading and using a pre-built dataset from Pinecone Datasets. 

### 3. Simple RAG (simple_rag.ipynb)

* Offers a basic implementation of a Retrieval-Augmented Generation (RAG) model.
* Provides a step-by-step guide on setting up and using the RAG model.
* Includes examples of answering questions based on retrieved documents. 

### 4. The Ultimate Guide on RAG with Gemma & Llama Index (the-ultimate-guide-on-rag-w-gemma-llama-index.ipynb)

* Provides a comprehensive guide to RAG using the Gemma language model and Llama Index.
* Covers key concepts like indexing, retrieval, and generation.
* Demonstrates building a chatbot capable of:
    - Understanding and explaining data science concepts.
    - Finding and summarizing research papers.
    - Exploring papers on specific topics.
    - Suggesting related concepts for further exploration.

### 5. Langchain Llama3 (langchain-llama3)

This subdirectory contains notebooks exploring RAG using Langchain and the LLaMA3 model.

**5.1 Web (langchain-llama3/web.ipynb)**

  * Demonstrates using Langchain components like:
     - `WebResearchRetriever` for web research.
     - `Chroma` for vector storage.
     - `GoogleSearchAPIWrapper` for querying Google Search.
     - `OllamaEmbeddings` and `Ollama` for embeddings and LLM interaction, respectively.
  * Showcases loading documents using `WikipediaLoader`.

**5.2 Langgraph RAG Agent (langchain-llama3/langgraph_rag_agent_llama3_local.ipynb)**
  * Combines various RAG techniques into a single agent:
     - **Routing (Adaptive RAG):** Routes questions to different retrieval approaches.
     - **Fallback (Corrective RAG):** Uses web search if local documents are not relevant.
     - **Self-correction (Self-RAG):** Attempts to fix hallucinations or incomplete answers.
  * Demonstrates using GPT4All embeddings and the Ollama LLM.

## Getting Started

1. Clone the repository: `git clone https://github.com/
