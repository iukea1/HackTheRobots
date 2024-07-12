# Import necessary libraries
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
import requests, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09")

# Tool 1: Save the news articles in a database
class SearchNewsDB:
    @tool("News DB Tool")
    def news(self, query: str):
        """Fetch news articles and process their contents."""
        API_KEY = os.getenv('NEWSAPI_KEY')  # Fetch API key from environment variable
        if not API_KEY:
            raise ValueError("NEWSAPI_KEY environment variable not set.")
        
        base_url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'sortBy': 'publishedAt',
            'apiKey': API_KEY,
            'language': 'en',
            'pageSize': 5,
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            error_message = response.json().get('message', 'Unknown error occurred.')
            return f"Failed to retrieve news. Error: {error_message}"
        
        articles = response.json().get('articles', [])
        all_splits = []
        for article in articles:
            loader = WebBaseLoader(article['url'])
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
        
        if all_splits:
            vectorstore = Chroma.from_documents(all_splits, embedding=embedding_function, persist_directory="./chroma_db")
            retriever = vectorstore.similarity_search(query)
            return retriever
        else:
            return "No content available for processing."

# Tool 2: Get the news articles from the database
class GetNews:
    @tool("Get News Tool")
    def news(self, query: str) -> str:
        """Search Chroma DB for relevant news information based on a query."""
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        retriever = vectorstore.similarity_search(query)
        return retriever

# Tool 3: Search for news articles on the web
search_tool = DuckDuckGoSearchRun()

# 2. Creating Agents
news_search_agent = Agent(
    role='News Seacher',
    goal='Generate key points for each news article from the latest news',
    backstory='Expert in analysing and generating key points from news content for quick updates.',
    tools=[SearchNewsDB().news],
    allow_delegation=True,
    verbose=True,
    llm=llm
)

writer_agent = Agent(
    role='Writer',
    goal='Identify all the topics received. Use the Get News Tool to verify each topic to search. Use the Search tool for detailed exploration of each topic. Summarise the retrieved information in depth for every topic.',
    backstory='Expert in crafting engaging narratives from complex information.',
    tools=[GetNews().news, search_tool],
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# 3. Creating Tasks
news_search_task = Task(
    description='Search for AI 2024 and create key points for each news.',
    agent=news_search_agent,
    tools=[SearchNewsDB().news],
    expected_output="List of key points for each news article"  # Example expected output
)

writer_task = Task(
    description="""
    Go step by step.
    Step 1: Identify all the topics received.
    Step 2: Use the Get News Tool to verify each topic by going through one by one.
    Step 3: Use the Search tool to search for information on each topic one by one. 
    Step 4: Go through every topic and write an in-depth summary of the information retrieved.
    Don't skip any topic.
    """,
    agent=writer_agent,
    tools=[GetNews().news, search_tool],
    expected_output="In-depth summaries for each topic"  # Example expected output
)

# 4. Creating Crew
# Define your crew and processes as needed

# 4. Creating Crew
news_crew = Crew(
    agents=[news_search_agent, writer_agent],
    tasks=[news_search_task, writer_task],
    process=Process.sequential, 
    manager_llm=llm
)

# Execute the crew to see RAG in action
result = news_crew.kickoff()
print(result)