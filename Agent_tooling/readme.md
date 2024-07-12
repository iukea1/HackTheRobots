# Agent Tooling Notebooks

This directory contains Jupyter notebooks showcasing different techniques and use cases for building and utilizing agents in Langchain.

## Folder Structure:

The notebooks are organized into the following subfolders:

* **Fundamentals:** For notebooks demonstrating core agent concepts (e.g., async execution, error handling).
* **Integrations:** For notebooks showcasing agent interaction with external tools and databases.
* **Applications:** For notebooks focused on specific use cases and agent-powered applications.


## Fundamentals

* **async_agent.ipynb:** Demonstrates how to initialize and run an agent asynchronously using the `asyncio` library, highlighting the performance benefits of concurrent execution.
* **async_llm.ipynb:** Provides a basic example of using an LLM (Large Language Model) asynchronously for faster processing of multiple requests.
* **handle_parsing_errors.ipynb:** Shows how to handle parsing errors that occur when an agent's output is not formatted correctly for the output parser. It includes using default error handling, custom error messages, and custom error functions.

## Integrations

* **neo4j_llama_multimodal.ipynb:** Demonstrates building a multimodal agent that combines text and image data stored in a Neo4j graph database. It covers data loading, processing, embedding, and querying.
* **web_scraping.ipynb:**  Explores various techniques for web scraping using agents. 
    * Covers basic scraping with Beautiful Soup.
    * Demonstrates LLM-powered extraction chains for adaptive scraping. 
    * Introduces the `WebResearchRetriever` for automating web research and answering questions over websites.
    * Shows an example of using Apify's `Website Content Crawler` to crawl and answer questions over a specific website's documentation. 
  * **sitemap.ipynb:**  This notebook demonstrates how to use the Sitemap Loder to load a sitemap from a URL and scrape all linked pages.  It covers
   * Loading sitemaps both locally and from the web
   * Filtering large sitemaps 
   * Adding custom scraping rules to the process


## Applications

* **Job_search_agents/app.ipynb:**  Presents a real-world example of building a job search agent. It showcases creating specialized agents (News Searcher, Writer) and coordinating them within a `Crew` to automate information gathering and summarization from news articles. 


## Getting Started

To run these notebooks, make sure you have the necessary dependencies installed. Each notebook usually includes an installation cell at the beginning. 

## Contributing

Contributions are welcome! If you have new agent tooling examples, feel free to create a pull request. Please follow the contribution guidelines outlined in the repository.
