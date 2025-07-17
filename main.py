# main.py

from pydantic import BaseModel
from typing import Literal
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_react_agent
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from sqlalchemy import text
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class AgentOutput(BaseModel):
    """Base class for LangGraph Workflow output DTO, compatible with A2A protocol"""
    task_status: Literal["completed", "input_required", "error"]
    response: str


if __name__ == "__main__":
    # Initialize Azure-hosted OpenAI chat model from .env variables
    # llm = AzureChatOpenAI(
    #     deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    #     openai_api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
    #     openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    #     openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    #     temperature=0,
    # )

    # # Initialize SQLite database and SQLDatabaseToolkit
    # db = SQLDatabase.from_uri(f"sqlite:///{Path(__file__).parent}/Chinook.db")
    # db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # tools = db_toolkit.get_tools()

    # system_prompt = """
    # You are an agent designed to interact with a SQL database.
    # Given an input question, create a syntactically correct {dialect} query to run,
    # then look at the results of the query and return the answer. Unless the user
    # specifies a specific number of examples they wish to obtain, always limit your
    # query to at most {top_k} results.

    # You can order the results by a relevant column to return the most interesting
    # examples in the database. Never query for all the columns from a specific table,
    # only ask for the relevant columns given the question.

    # You MUST double check your query before executing it. If you get an error while
    # executing a query, rewrite the query and try again.

    # DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    # database.

    # To start you should ALWAYS look at the tables in the database to see what you
    # can query. Do NOT skip this step.

    # Then you should query the schema of the most relevant tables.
    # """.format(
    #         dialect=db.dialect,
    #         top_k=5,
    #     )

    # graph = create_react_agent(
    #     model=llm,
    #     tools=tools,
    #     prompt=system_prompt,
    #     response_format=AgentOutput
    # )

    # Replace user input with a hardcoded SQL query
    query = "SELECT Name FROM Artist LIMIT 3"
    print(f"Executing hardcoded query: {query}")

    # Execute the SQL query directly
    db = SQLDatabase.from_uri(f"sqlite:///{Path(__file__).parent}/Chinook.db")
    engine = db._engine
    with engine.connect() as conn:
        result = conn.execute(text(query))
        print("Query Results:")
        for row in result:
            print(row)