# SQL Agent

This agent is designed to interact with a SQL database (Chinook.db) using natural language queries.
It leverages Azure OpenAI for SQL query generation and execution.

## Functionality

- Translates natural language questions into SQL queries.
- Executes SQL queries against the local `Chinook.db` database.
- Returns query results in a readable format.

## Setup & Dependencies

This agent requires the following environment variables to connect to Azure OpenAI:
- `AZURE_OPENAI_DEPLOYMENT_NAME`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_API_KEY`

It utilizes the `langchain-openai` and `langchain-community` libraries for LLM interaction and SQL toolkit functionality.
The `Chinook.db` SQLite database file must be present in the agent's main directory.

## Usage

Once deployed via the AGNTCY Workflow Server Manager (WFSM), this agent can be invoked via the `/runs` API endpoint by providing a natural language `query`.

**Example Query:**
`{"query": "What are the top 5 artists by total sales?"}`
