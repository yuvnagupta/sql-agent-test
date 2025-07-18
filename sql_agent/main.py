# sql_agent/main.py

import asyncio
import os
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
# Removed: from langchain_community.utilities import SQLDatabase
# Removed: from sqlalchemy import text
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from pydantic import BaseModel
# Removed: from langchain_community.agent_toolkits import SQLDatabaseToolkit
# Removed: from langchain_core.agents import AgentAction, AgentFinish
# Removed: from langchain_core.messages import HumanMessage

# --- LangGraph specific imports ---
from langgraph.graph import StateGraph, END # Use END for a simple direct path
#from langgraph.graph.graph import CompiledGraph
from typing_extensions import TypedDict

# Load environment variables from .env file
load_dotenv()

class AgentOutput(BaseModel):
    """Base class for LangGraph Workflow output DTO, compatible with A2A protocol"""
    task_status: Literal["completed", "input_required", "error"]
    response: str

# --- Graph State Definition ---
class AgentState(TypedDict):
    """Represents the state of our LangGraph agent."""
    query: str
    response: str
    error: str

# --- Agent Initialization (Module-level for Workflow Server) ---

llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
)

# --- IN-MEMORY TOY DATABASE ---
# This replaces the Chinook.db connection
TOY_DATABASE_SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    city TEXT
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    price REAL,
    stock INTEGER
);

-- Sample Data (for reference, actual data handled in the code below)
INSERT INTO users (id, name, age, city) VALUES (1, 'Alice', 30, 'New York');
INSERT INTO users (id, name, age, city) VALUES (2, 'Bob', 24, 'Los Angeles');
INSERT INTO users (id, name, age, city) VALUES (3, 'Charlie', 35, 'New York');
INSERT INTO products (product_id, product_name, price, stock) VALUES (101, 'Laptop', 1200.00, 50);
INSERT INTO products (product_id, product_name, price, stock) VALUES (102, 'Mouse', 25.00, 200);
"""

# Simple in-memory data for the toy database
TOY_DATABASE_DATA = {
    "users": [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "city": "Los Angeles"},
        {"id": 3, "name": "Charlie", "age": 35, "city": "New York"},
    ],
    "products": [
        {"product_id": 101, "product_name": "Laptop", "price": 1200.00, "stock": 50},
        {"product_id": 102, "product_name": "Mouse", "price": 25.00, "stock": 200},
    ]
}

# --- Mock SQL Execution Function ---
# This simulates executing SQL queries against the toy data
def execute_toy_sql(sql_query: str) -> str:
    """A very basic mock SQL executor for the toy database."""
    sql_query = sql_query.lower().strip()

    # Simple SELECT * FROM users/products logic for demonstration
    if "select * from users" in sql_query:
        headers = ["id", "name", "age", "city"]
        data = TOY_DATABASE_DATA["users"]
    elif "select * from products" in sql_query:
        headers = ["product_id", "product_name", "price", "stock"]
        data = TOY_DATABASE_DATA["products"]
    elif "select name from users" in sql_query or "select * from users limit" in sql_query:
        # Example for specific column/limit queries
        headers = ["name"]
        data = [{"name": u["name"]} for u in TOY_DATABASE_DATA["users"]]
        if "limit 1" in sql_query: data = data[:1]
        elif "limit 2" in sql_query: data = data[:2]
        elif "limit 3" in sql_query: data = data[:3]
    elif "select product_name from products" in sql_query:
        headers = ["product_name"]
        data = [{"product_name": p["product_name"]} for p in TOY_DATABASE_DATA["products"]]
    else:
        return "ERROR: Query not supported by toy database mock, or invalid SQL."

    # Format output as a simple table string
    header_str = "\t".join(headers)
    rows_str = "\n".join(["\t".join(map(str, row.values())) for row in data])
    return f"{header_str}\n{rows_str}"


# --- AgentOutput is now used as the state for the workflow ---

# --- LangGraph Node ---
async def generate_and_execute_sql(state: AgentOutput) -> AgentOutput:
    """
    LangGraph node: Takes a natural language query and outputs the answer based only on the simple table in the system prompt.
    """
    query = state.response  # We'll use the 'response' field to carry the query

    # Define a simple table (3 rows, 3 columns) as a string for the system prompt
    table = (
        "| id | name   | age | city        |\n"
        "|----|--------|-----|-------------|\n"
        "| 1  | Alice  | 30  | New York    |\n"
        "| 2  | Bob    | 24  | Los Angeles |\n"
        "| 3  | Charlie| 35  | New York    |"
    )

    prompt = f"""
You are a helpful assistant. Here is a table you can use to answer questions:\n\n{table}\n\nAnswer the following question using only the information in the table above. If the answer is not present, say 'I don't know'.\n\nQuestion: {query}\nAnswer:\n"""
    try:
        llm_response = llm.invoke(prompt)
        answer = llm_response.content.strip()
        return AgentOutput(task_status="completed", response=answer)
    except Exception as e:
        return AgentOutput(task_status="error", response=f"LLM failed: {str(e)}")

# --- LangGraph Definition ---
workflow = StateGraph(AgentOutput)

# Add the single node that performs SQL generation and execution
workflow.add_node("sql_executor_node", generate_and_execute_sql) # Renamed node for clarity

# Set the entry point and connect it directly to the END
workflow.set_entry_point("sql_executor_node")
workflow.add_edge("sql_executor_node", END) # Simple direct path: Node -> End

# Compile the graph into a runnable instance
sql_graph_runnable = workflow.compile() # This is the compiled graph

# --- Define the runnable entry point for the Workflow Server ---
async def sql_agent_runnable(input_data: Dict[str, Any]) -> AgentOutput:
    """
    Runnable wrapper for the SQL Agent to integrate with the Workflow Server.
    input_data is expected to contain a 'query' field.
    """
    query = input_data.get("query")
    if not query:
        return AgentOutput(task_status="error", response="Input missing 'query' field.")

    # Initialize the graph state with the incoming query in the response field
    initial_state = AgentOutput(task_status="input_required", response=query)

    try:
        # Invoke the compiled LangGraph with the initial state.
        final_state = await sql_graph_runnable.ainvoke(initial_state)

        # Determine the final output status and response based on the graph's final state
        if final_state.get("error"):
            return AgentOutput(task_status="error", response=final_state["error"])
        elif final_state.get("response"):
            return AgentOutput(task_status="completed", response=final_state["response"])
        else:
            return AgentOutput(task_status="error", response="Agent completed without a clear response or error.")

    except Exception as e:
        return AgentOutput(task_status="error", response=f"LangGraph execution failed: {str(e)}")

# --- Local Testing Block ---
if __name__ == "__main__":
    print("--- Testing sql_agent_runnable locally ---")

    # Test case 1: Simple non-database query (direct response)
    async def test_simple_query():
        local_input = {"query": "What is the purpose of this agent?"}
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (Simple Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_simple_query())

    # Test case 2: SQL query for toy database
    async def test_sql_query():
        local_input = {"query": "SELECT name FROM users LIMIT 1;"}
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (SQL Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_sql_query())

    # Test case 3: Another SQL query for toy database
    async def test_another_sql_query():
        local_input = {"query": "SELECT product_name, price FROM products WHERE stock > 100;"}
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (Another SQL Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_another_sql_query())

    # Test case 4: Unsupported SQL query for toy database
    async def test_unsupported_sql_query():
        local_input = {"query": "SELECT * FROM orders;"} # This table doesn't exist in toy DB
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (Unsupported SQL Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_unsupported_sql_query())
