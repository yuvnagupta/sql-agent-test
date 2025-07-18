# sql_agent/main.py

import asyncio
import os
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from pydantic import BaseModel
from sqlalchemy import text

# Load environment variables from .env file
load_dotenv()

class AgentOutput(BaseModel):
    """Base class for LangGraph Workflow output DTO, compatible with A2A protocol"""
    task_status: Literal["completed", "input_required", "error"]
    response: str

# --- Agent Initialization (Module-level for Workflow Server) ---

# Initialize Azure-hosted OpenAI chat model from environment variables
# This will pick up AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, etc.
# from your Docker Compose environment or local .env file.
llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
)

# Initialize SQLite database
# Ensure Chinook.db is in the same directory as this main.py file
db = SQLDatabase.from_uri(f"sqlite:///{Path(__file__).parent}/Chinook.db")

# --- Simple LLM-to-Table QA Function ---
SMALL_TABLE = '''
| Name     | Age | City      |
|----------|-----|-----------|
| Alice    | 30  | New York  |
| Bob      | 25  | Chicago   |
| Charlie  | 35  | San Diego |
'''

async def table_qa_agent(input_data: Dict[str, Any]) -> AgentOutput:
    """
    Simple QA: LLM answers questions about a small table (no SQL).
    input_data is expected to contain a 'query' field.
    """
    query = input_data.get("query")
    if not query:
        return AgentOutput(task_status="error", response="Input missing 'query' field.")
    prompt = f"""
You are a helpful assistant. Here is a table:
{SMALL_TABLE}
Answer the following question using only the data in the table. Be concise.
Question: {query}
Answer:
"""
    try:
        llm_response = llm.invoke(prompt)
        return AgentOutput(task_status="completed", response=llm_response.content.strip())
    except Exception as e:
        return AgentOutput(task_status="error", response=f"Failed: {str(e)}")

# --- Simple LLM-to-SQL Function ---
CHINOOK_SCHEMA = '''
CREATE TABLE [Album]( [AlbumId] INTEGER NOT NULL, [Title] NVARCHAR(160) NOT NULL, [ArtistId] INTEGER NOT NULL, PRIMARY KEY ([AlbumId]), FOREIGN KEY ([ArtistId]) REFERENCES [Artist] ([ArtistId]));
CREATE TABLE [Artist]( [ArtistId] INTEGER NOT NULL, [Name] NVARCHAR(120), PRIMARY KEY ([ArtistId]));
CREATE TABLE [Customer]( [CustomerId] INTEGER NOT NULL, [FirstName] NVARCHAR(40) NOT NULL, [LastName] NVARCHAR(20) NOT NULL, [Company] NVARCHAR(80), [Address] NVARCHAR(70), [City] NVARCHAR(40), [State] NVARCHAR(40), [Country] NVARCHAR(40), [PostalCode] NVARCHAR(10), [Phone] NVARCHAR(24), [Fax] NVARCHAR(24), [Email] NVARCHAR(60) NOT NULL, [SupportRepId] INTEGER, PRIMARY KEY ([CustomerId]), FOREIGN KEY ([SupportRepId]) REFERENCES [Employee] ([EmployeeId]));
CREATE TABLE [Employee]( [EmployeeId] INTEGER NOT NULL, [LastName] NVARCHAR(20) NOT NULL, [FirstName] NVARCHAR(20) NOT NULL, [Title] NVARCHAR(30), [ReportsTo] INTEGER, [BirthDate] DATETIME, [HireDate] DATETIME, [Address] NVARCHAR(70), [City] NVARCHAR(40), [State] NVARCHAR(40), [Country] NVARCHAR(40), [PostalCode] NVARCHAR(10), [Phone] NVARCHAR(24), [Fax] NVARCHAR(24), [Email] NVARCHAR(60), PRIMARY KEY ([EmployeeId]), FOREIGN KEY ([ReportsTo]) REFERENCES [Employee] ([EmployeeId]));
CREATE TABLE [Genre]( [GenreId] INTEGER NOT NULL, [Name] NVARCHAR(120), PRIMARY KEY ([GenreId]));
CREATE TABLE [Invoice]( [InvoiceId] INTEGER NOT NULL, [CustomerId] INTEGER NOT NULL, [InvoiceDate] DATETIME NOT NULL, [BillingAddress] NVARCHAR(70), [BillingCity] NVARCHAR(40), [BillingState] NVARCHAR(40), [BillingCountry] NVARCHAR(40), [BillingPostalCode] NVARCHAR(10), [Total] NUMERIC(10,2) NOT NULL, PRIMARY KEY ([InvoiceId]), FOREIGN KEY ([CustomerId]) REFERENCES [Customer] ([CustomerId]));
CREATE TABLE [InvoiceLine]( [InvoiceLineId] INTEGER NOT NULL, [InvoiceId] INTEGER NOT NULL, [TrackId] INTEGER NOT NULL, [UnitPrice] NUMERIC(10,2) NOT NULL, [Quantity] INTEGER NOT NULL, PRIMARY KEY ([InvoiceLineId]), FOREIGN KEY ([InvoiceId]) REFERENCES [Invoice] ([InvoiceId]), FOREIGN KEY ([TrackId]) REFERENCES [Track] ([TrackId]));
CREATE TABLE [MediaType]( [MediaTypeId] INTEGER NOT NULL, [Name] NVARCHAR(120), PRIMARY KEY ([MediaTypeId]));
CREATE TABLE [Playlist]( [PlaylistId] INTEGER NOT NULL, [Name] NVARCHAR(120), PRIMARY KEY ([PlaylistId]));
CREATE TABLE [PlaylistTrack]( [PlaylistId] INTEGER NOT NULL, [TrackId] INTEGER NOT NULL, PRIMARY KEY ([PlaylistId], [TrackId]), FOREIGN KEY ([PlaylistId]) REFERENCES [Playlist] ([PlaylistId]), FOREIGN KEY ([TrackId]) REFERENCES [Track] ([TrackId]));
CREATE TABLE [Track]( [TrackId] INTEGER NOT NULL, [Name] NVARCHAR(200) NOT NULL, [AlbumId] INTEGER, [MediaTypeId] INTEGER NOT NULL, [GenreId] INTEGER, [Composer] NVARCHAR(220), [Milliseconds] INTEGER NOT NULL, [Bytes] INTEGER, [UnitPrice] NUMERIC(10,2) NOT NULL, PRIMARY KEY ([TrackId]), FOREIGN KEY ([AlbumId]) REFERENCES [Album] ([AlbumId]), FOREIGN KEY ([GenreId]) REFERENCES [Genre] ([GenreId]), FOREIGN KEY ([MediaTypeId]) REFERENCES [MediaType] ([MediaTypeId]));
'''

# async def sql_agent_runnable(input_data: Dict[str, Any]) -> AgentOutput:
#     """
#     Simple wrapper: LLM writes SQL, then we execute it and return the result.
#     input_data is expected to contain a 'query' field.
#     """
#     query = input_data.get("query")
#     if not query:
#         return AgentOutput(task_status="error", response="Input missing 'query' field.")
#
#     # Prompt LLM to write a SQL query for the user's question, with schema
#     prompt = f"""
# You are an expert SQL assistant. Here is the schema of the database you will query:
# {CHINOOK_SCHEMA}
# Given the following question, write a syntactically correct SQLite SQL query. Do not explain, just output the SQL query.
# Question: {query}
# SQL Query:
# """
#     try:
#         llm_response = llm.invoke(prompt)
#         sql_text = llm_response.content.strip()
#         # Remove code block markers if present
#         if sql_text.startswith("```sql"):
#             sql_text = sql_text[6:]
#         if sql_text.startswith("```"):
#             sql_text = sql_text[3:]
#         if sql_text.endswith("```"):
#             sql_text = sql_text[:-3]
#         sql_query = sql_text.strip()
#         # Execute the SQL query
#         with db._engine.connect() as conn:
#             result = conn.execute(text(sql_query))
#             rows = result.fetchall()
#             columns = result.keys()
#             # Format output as a table
#             output = "\t".join(columns) + "\n" + "\n".join(["\t".join(map(str, row)) for row in rows])
#         return AgentOutput(task_status="completed", response=output)
#     except Exception as e:
#         return AgentOutput(task_status="error", response=f"Failed: {str(e)}")

# --- Local Testing Block ---
if __name__ == "__main__":
    print("--- Testing table_qa_agent locally ---")
    local_input = {"query": "Who is from Chicago?"}
    async def run_local_test():
        result = await table_qa_agent(local_input)
        print("\nLocal Test Result:")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(run_local_test())

    # You can add more local tests here:
    # local_input_2 = {"query": "Show me the total sales for each customer."}
    # asyncio.run(sql_agent_runnable(local_input_2))