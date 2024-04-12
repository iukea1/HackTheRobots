# RAGs Querying Notebooks

Welcome to the RAGs Querying Notebooks repository. This repository contains a collection of Jupyter notebooks that demonstrate the implementation and usage of Retrieval-Augmented Generation (RAG) models for querying tasks.

## Project Structure

The project is structured into several notebooks, each focusing on a different aspect of querying with RAG models:

1. [Multi-Document Agents](Multi_Document_Agents.ipynb): This notebook demonstrates how to use the `MultiDocumentAgent` class to handle complex queries that span across multiple documents. It shows how to initialize a `MultiDocumentAgent`, how to add documents to it, and how to use it to answer queries.

2. [SubQuestion Query Engine](SubQuestion_Query_Engine.ipynb): This notebook delves into addressing complex queries that extend over various documents by breaking them down into simpler sub-queries and generate answers using the `SubQuestionQueryEngine`.

3. [App](app.py): This Python script provides a Gradio interface for processing academic papers and performing queries on them. It uses the ArXiv API to download papers based on a search query, processes the papers and saves them to an embeddings database, and then allows the user to perform queries on the processed papers.

## Getting Started

To begin exploring this project:

1. Clone the repository to your local machine.
2. Navigate to the RAGs/querying directory.
3. Follow the instructions in each notebook to install the necessary dependencies.

## Contributing

We welcome contributions to this project. Whether it's improving the existing notebooks, adding new ones, or providing feedback, your contributions are greatly appreciated. Please see the contributing guidelines for more information.