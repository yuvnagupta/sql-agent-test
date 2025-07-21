import asyncio
import os
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from pydantic import BaseModel
import sqlite3
# SQL database functionality
from langchain_community.utilities import SQLDatabase

# --- LangGraph specific imports ---
from langgraph.graph import StateGraph, END
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
    task_status: str

# --- Agent Initialization (Module-level for Workflow Server) ---

llm = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
)

# --- ClickHouse SQLite Database Connection ---
CLICKHOUSE_DB_PATH = os.path.join(os.path.dirname(__file__), "clickhouse.db")
db = SQLDatabase.from_uri(f"sqlite:///{CLICKHOUSE_DB_PATH}")

# --- Database Schema and Description ---
CLICKHOUSE_SCHEMA = """
CREATE TABLE llm_usage (
    request_date TEXT,
    cluster TEXT,
    namespace TEXT,
    tenant_id TEXT,
    bot_id INTEGER,
    llm_provider TEXT,
    model TEXT,
    prompt_name TEXT,
    trace_id TEXT,
    duration_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER
)
"""

CLICKHOUSE_DESCRIPTION = """
The llm_usage table contains information about LLM API calls:
- request_date: When the request was made (timestamp)
- cluster: The compute cluster that processed the request
- namespace: The namespace within the cluster
- tenant_id: Customer/organization identifier
- bot_id: Unique identifier for the bot
- llm_provider: Provider of the LLM (e.g., OpenAI, Anthropic, Mistral)
- model: The specific model used (e.g., GPT-4, Claude-2.1)
- prompt_name: Type of prompt used
- trace_id: Unique identifier for the request for tracing
- duration_ms: Request processing time in milliseconds
- prompt_tokens: Number of tokens in the prompt
- completion_tokens: Number of tokens in the completion
- total_tokens: Total tokens used (prompt + completion)
"""

# --- SQL Query Functions ---
def get_schema_str():
    """Get the database schema as a string."""
    return db.get_table_info()

def run_sql_query(query: str) -> str:
    """Run a SQL query against the ClickHouse database and return results as a string."""
    try:
        result = db.run(query)
        return result
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

# --- LangGraph Node ---
async def generate_and_execute_sql(state: AgentState) -> AgentState:
    """
    LangGraph node: Takes a natural language query and generates/executes SQL against the ClickHouse database.
    """
    query = state.get("query", "")
    
    try:
        schema = get_schema_str()
        
        # Create a prompt that instructs the LLM to generate SQL
        sql_generation_prompt = f"""
You are an expert SQL assistant. Given a question, create a syntactically correct SQLite query to answer it.

Database schema:
{schema}

Database description:
{CLICKHOUSE_DESCRIPTION}

Given the question, write a syntactically correct SQLite SQL query. Do not explain, just output the SQL query.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.

Question: {query}
SQL Query:
"""
        # Generate SQL using LLM
        sql_response = llm.invoke(sql_generation_prompt)
        sql_query = sql_response.content.strip()
        
        # Clean up SQL query (remove markdown code blocks if present)
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        # Execute the SQL query
        query_result = run_sql_query(sql_query)
            
        # Generate a natural language response using the query result
        response_prompt = f"""
You are a helpful database assistant. Below is a question and the result of a SQL query that answers the question.
Please provide a clear, concise response to the question using the SQL query result.

Question: {query}

SQL Query:
{sql_query}

Query Result:
{query_result}

Response:
"""
        final_response = llm.invoke(response_prompt).content
        
        return {
            "query": query,
            "response": final_response,
            "error": "",
            "task_status": "completed"
        }
    except Exception as e:
        return {
            "query": query,
            "response": "",
            "error": f"Error: {str(e)}",
            "task_status": "error"
        }

# --- LangGraph Definition ---
workflow = StateGraph(AgentState)

# Add the single node that performs SQL generation and execution
workflow.add_node("sql_executor_node", generate_and_execute_sql)

# Set the entry point and connect it directly to the END
workflow.set_entry_point("sql_executor_node")
workflow.add_edge("sql_executor_node", END)

# Compile the graph into a runnable instance
sql_graph_runnable = workflow.compile()

# --- Define the runnable entry point for the Workflow Server ---
async def sql_agent_runnable(input_data: Dict[str, Any]) -> AgentOutput:
    """
    Runnable wrapper for the SQL Agent to integrate with the Workflow Server.
    input_data is expected to contain a 'query' field.
    """
    query = input_data.get("query")
    if not query:
        return AgentOutput(task_status="error", response="Input missing 'query' field.")

    # Initialize the graph state with the incoming query
    initial_state = {
        "query": query,
        "response": "",
        "error": "",
        "task_status": "input_required"
    }

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

    # Test case 1: Simple database information query
    async def test_simple_query():
        local_input = {"query": "What is the schema of the database?"}
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (Schema Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_simple_query())

    # Test case 2: LLM provider analysis
    async def test_llm_providers_query():
        local_input = {"query": "What are the top 5 LLM providers by total token usage?"}
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (LLM Providers Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_llm_providers_query())

    # Test case 3: Model-specific query
    async def test_model_query():
        local_input = {"query": "What is the average response time for Claude-2.1 model?"}
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (Model Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_model_query())

    # Test case 4: Tenant analysis
    async def test_tenant_query():
        local_input = {"query": "Show me the top 3 tenants with the highest token usage."}
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (Tenant Analysis Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_tenant_query())

    # Test case 5: Prompt type analysis
    async def test_prompt_query():
        local_input = {"query": "What prompt types are used and which one has the highest average completion tokens?"}
        result = await sql_agent_runnable(local_input)
        print("\nLocal Test Result (Prompt Analysis Query):")
        print(f"Task Status: {result.task_status}")
        print(f"Response: {result.response}")
    asyncio.run(test_prompt_query())
