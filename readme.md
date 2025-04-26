# Customer Service Agent with AutoGen and Streamlit

This AI-powered customer service system consists of a Streamlit-based frontend and a FastAPI backend, designed to handle customer inquiries efficiently. The frontend provides an interactive chat interface, while the backend coordinates two AI agents: a general agent for common queries and a senior agent for escalated or complex issues. Both agents utilize Azure OpenAI models and search respective CSV-based knowledge bases using fuzzy matching to provide relevant answers. The system ensures smooth escalation between agents, logs interactions, and handles errors gracefully, offering a robust solution for retail customer service scenarios.

## Features

- **General Agent**: Handles common customer queries using a general knowledge base. It is powered by the GPT-4o model from Azure OpenAI Service.
- **Senior Agent**: Handles escalated or complex queries using a senior knowledge base. It is powered by the o3-mini model from Azure OpenAI Service.
- **Streamlit Frontend**: Provides an interactive chat interface for users.
- **FastAPI Backend**: Manages the logic for agent interactions and knowledge base lookups.
- **Knowledge Base Search**: Uses fuzzy matching to find relevant answers in CSV-based knowledge bases.

## Project Structure

```
customer_service_agent_with_autogen_streamlit/
├── backend_api.py       # FastAPI backend for agent interactions
├── frontend_app.py      # Streamlit frontend for user interaction
├── general_agent.csv    # General knowledge base
├── senior_agent.csv     # Senior knowledge base
├── requirement.txt      # List of dependencies required for the project.
├── readme.md            # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI API credentials
- Required Python packages (see below)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd customer_service_agent_with_autogen_streamlit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Azure OpenAI API credentials:
   - Add your API key and endpoint to environment variables:
     ```bash
     export AZURE_OPENAI_API_KEY="<your-api-key>"
     export AZURE_OPENAI_ENDPOINT="<your-endpoint>"
     ```

4. Ensure the knowledge base files (`general_agent.csv` and `senior_agent.csv`) are in the project directory.

## Usage

### Start the Backend

Run the FastAPI backend:
```bash
python backend_api.py
```
The backend will start at `http://127.0.0.1:8000`.

### Start the Frontend

Run the Streamlit frontend:
```bash
streamlit run frontend_app.py
```
The frontend will open in your default web browser.

### Interact with the Agents

1. Open the Streamlit app in your browser.
2. Type your question in the chat input box.
3. The system will respond using the general or senior agent, depending on the query.

## Knowledge Base Format

The knowledge base files (`general_agent.csv` and `senior_agent.csv`) should have the following columns:

- `Question`: The customer query.
- `Answer`: The corresponding answer.

## Agent Logic

The system uses a multi-agent architecture to handle customer queries effectively. The agents are defined as follows:

### General Agent
- **Purpose**: Handles common customer queries using the general knowledge base.
- **Logic**:
  1. Searches the general knowledge base for an answer using the `retrieve_from_general_kb` function.
  2. If a relevant answer is found, it is provided directly to the customer.
  3. If no answer is found or the query involves sensitive topics, the agent escalates the query to the senior agent.

### Senior Agent
- **Purpose**: Handles escalated or complex queries using the senior knowledge base.
- **Logic**:
  1. Searches the senior knowledge base for an answer using the `retrieve_from_senior_kb` function.
  2. If a relevant answer is found, it is provided directly to the customer.
  3. If no answer is found, the agent uses its expertise to provide a comprehensive response.

### User Proxy Agent
- **Purpose**: Acts as an interface between the user and the system.
- **Logic**:
  1. Receives user input and initiates the chat process.
  2. Manages the flow of messages between the user and the agents.

### Group Chat
- **Purpose**: Coordinates interactions between the agents and manages the chat history.
- **Logic**:
  1. Maintains a list of messages exchanged during the chat.
  2. Determines when to terminate the chat based on predefined conditions.

### Termination Conditions
- The chat terminates when:
  1. The general agent provides a direct answer (not an escalation).
  2. The senior agent provides a substantial response to an escalated query.

## Troubleshooting

- **Knowledge Base Not Found**: Ensure `general_agent.csv` and `senior_agent.csv` are in the project directory.
- **API Errors**: Verify your Azure OpenAI API credentials and endpoint.
- **Backend Connection Issues**: Ensure the FastAPI backend is running and accessible.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [AutoGen](https://github.com/microsoft/autogen) for the agent framework.
- [Streamlit](https://streamlit.io/) for the frontend framework.
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.