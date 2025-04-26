import os
import logging
import pandas as pd
from fuzzywuzzy import fuzz
# Make sure all necessary components are imported
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, Agent
from typing import Dict, Union, Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json # Import json for error handling

# --- Configuration and Setup ---
# ... (keep existing setup code: logging, KB loading, credentials, LLM configs) ...
# Disable Docker usage
os.environ["AUTOGEN_USE_DOCKER"] = "False"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load knowledge base CSV files (adjust paths if needed)
try:
    general_knowledge = pd.read_csv("general_agent.csv")
    senior_knowledge = pd.read_csv("senior_agent.csv")
except FileNotFoundError:
    logging.error("Knowledge base CSV files not found. Please ensure 'general_agent.csv' and 'senior_agent.csv' are in the same directory.")
    exit() # Exit if KBs are missing

# Define Azure OpenAI API deployment details and credentials (using environment variables is recommended)
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_API_KEY") # Replace or use env var
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "YOUR_AZURE_ENDPOINT") # Replace or use env var
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY") # Use env var if available
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR_AZURE_OPENAI_ENDPOINT.openai.azure.com") # Use env var if available
azure_deployment_general = "eason-gpt-4o" # Or your deployment name
azure_deployment_senior = "eason-o3-mini" # Or your deployment name
api_version = "2024-12-01-preview" # Or your API version

# Basic validation for credentials
if AZURE_OPENAI_API_KEY == "YOUR_AZURE_API_KEY" or AZURE_OPENAI_ENDPOINT == "YOUR_AZURE_ENDPOINT":
    logging.warning("Using placeholder API Key/Endpoint. Set environment variables AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")

# Define LLM configurations
llm_config_general = {
    "model": azure_deployment_general,
    "api_type": "azure",
    "api_version": api_version,
    "api_key": AZURE_OPENAI_API_KEY,
    "base_url": AZURE_OPENAI_ENDPOINT,
}

llm_config_senior = {
    "model": azure_deployment_senior,
    "api_type": "azure",
    "api_version": api_version,
    "api_key": AZURE_OPENAI_API_KEY,
    "base_url": AZURE_OPENAI_ENDPOINT,
}

# Define KB search function
def search_kb(query, kb_dataframe):
    query_str = str(query).lower()
    if 'Question' not in kb_dataframe.columns:
        logging.error("Knowledge base CSV must contain a 'Question' column.")
        return None, 0
    kb_dataframe['Question'] = kb_dataframe['Question'].astype(str)
    similar_questions = kb_dataframe['Question'].apply(lambda x: fuzz.ratio(x.lower(), query_str))
    if similar_questions.empty or similar_questions.isnull().all(): return None, 0
    max_score = similar_questions.max()
    if pd.notna(max_score) and max_score > 75:
        if 'Answer' not in kb_dataframe.columns:
             logging.error("Knowledge base CSV must contain an 'Answer' column.")
             return None, max_score
        answer = kb_dataframe.loc[similar_questions.idxmax(), 'Answer']
        return answer, max_score
    return None, max_score if pd.notna(max_score) else 0

# Define Tool Class
class CustomerServiceTools:
    @staticmethod
    def retrieve_from_general_kb(query):
        logging.info(f"Looking up in GENERAL KB: {query}")
        if isinstance(query, dict) and 'query' in query: query = query['query']
        answer, score = search_kb(query, general_knowledge)
        if answer:
            logging.info(f"Found general KB answer with score {score}")
            return str(answer)
        else:
            logging.info(f"No match found in general KB (best score: {score})")
            return "No answer found in general knowledge base."

    @staticmethod
    def retrieve_from_senior_kb(query):
        logging.info(f"Looking up in SENIOR KB: {query}")
        if isinstance(query, dict) and 'query' in query: query = query['query']
        answer, score = search_kb(query, senior_knowledge)
        if answer:
            logging.info(f"Found senior KB answer with score {score}")
            return str(answer)
        else:
            logging.info(f"No match found in senior KB (best score: {score})")
            return "No answer found in senior knowledge base."

tools = CustomerServiceTools()

# Define Tool Schemas
tool_schema_general_kb = {
    "type": "function", "function": {
        "name": "retrieve_from_general_kb",
        "description": "Search the GENERAL knowledge base for answers to common customer questions (store hours, basic returns, etc.).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The customer's question to look up."}}, "required": ["query"]}
    }
}
tool_schema_senior_kb = {
    "type": "function", "function": {
        "name": "retrieve_from_senior_kb",
        "description": "Search the SENIOR knowledge base for answers to complex or escalated issues (complaints, safety, policy exceptions, technical details).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The customer's complex/escalated question to look up."}}, "required": ["query"]}
    }
}

# --- Agent Definitions (ensure max_consecutive_auto_reply is appropriate) ---
user_proxy = UserProxyAgent(
    name="customer_api_interface",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False,
    function_map={
        "retrieve_from_general_kb": tools.retrieve_from_general_kb,
        "retrieve_from_senior_kb": tools.retrieve_from_senior_kb,
    }
)

general_agent = AssistantAgent(
    name="general_agent",
    llm_config={**llm_config_general, "tools": [tool_schema_general_kb]},
    system_message="""
You are a general customer service agent for a retail store.
For EVERY customer question:
1. FIRST, use the `retrieve_from_general_kb` function to search the general knowledge base.
2. If you get a relevant answer (NOT 'No answer found...'), provide it directly to the customer and conclude your turn.
3. If the general KB search returns 'No answer found...' OR the question involves sensitive topics (foreign objects, safety, complaints, disputes, policy exceptions, complex technical issues), you MUST reply ONLY with the exact phrase: 'I need to escalate this to our senior team.' and nothing else. This signals the escalation.

Do NOT attempt to answer sensitive topics yourself. Only answer directly if the general KB provides a clear, relevant answer AND the topic is NOT sensitive.
""",
    function_map={"retrieve_from_general_kb": tools.retrieve_from_general_kb},
    code_execution_config=False,
    max_consecutive_auto_reply=2 # Keep at 2
)

senior_agent = AssistantAgent(
    name="senior_agent",
    llm_config={**llm_config_senior, "tools": [tool_schema_senior_kb]},
    system_message="""
You are a senior customer service agent handling escalated issues. You ONLY speak AFTER the general_agent has stated 'I need to escalate this to our senior team.'

IMPORTANT: Your **FIRST and MANDATORY** action for EVERY escalated question you receive is to use the `retrieve_from_senior_kb` function. Do NOT generate any other response before using this function.

1.  **MANDATORY FIRST STEP:** Use the `retrieve_from_senior_kb` function to search the senior knowledge base using the customer's original query or the core issue described.
2.  **If the function returns a relevant answer:** Provide ONLY that information to the customer.
3.  **If the function returns 'No answer found...':** THEN (and ONLY then) use your expertise to analyze the situation and provide a comprehensive response, resolution, or clear next steps. Address complaints, disputes, technical matters, safety concerns, foreign objects, or policy exceptions with professionalism.
""",
    function_map={"retrieve_from_senior_kb": tools.retrieve_from_senior_kb},
    code_execution_config=False,
    max_consecutive_auto_reply=2 # Keep at 2
)

# --- Group Chat Setup ---
groupchat = GroupChat(
    agents=[user_proxy, general_agent, senior_agent],
    messages=[],
    max_round=10 # Keep max_round reasonable
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config_general
    # speaker_selection_method="round_robin" # Keep commented out for pyautogen v0.8.6
)

# --- Revised Termination Function (Focus on Sender and Content) ---
def is_termination_msg(message: Dict) -> bool:
    """Check if the message content indicates termination.
       Focuses on sender name and content due to role inconsistencies in logs.
    """
    sender_name = message.get("name", "")
    content = message.get("content", "")
    content_str = str(content).strip() if content is not None else ""

    # Terminate if general agent escalates (this seems reliable)
    if sender_name == general_agent.name and "I need to escalate this to our senior team." in content_str:
        logging.info(f"Termination condition met: Escalation by {sender_name}")
        return True

    # Terminate if senior agent provides a substantial response
    # Check if sender is senior_agent and content is not empty/None and longer than a short phrase
    if sender_name == senior_agent.name and content_str and len(content_str) > 50: # Adjust length threshold if needed
        logging.info(f"Termination condition met: Substantial response from {sender_name}")
        return True

    # Terminate if general agent provides a direct answer (not escalation)
    if sender_name == general_agent.name and content_str and "I need to escalate" not in content_str and content_str.endswith(('.', '!', '?')):
        logging.info(f"Termination condition met: Direct answer from {sender_name}")
        return True

    return False

# --- FastAPI Application ---
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    logging.info(f"Received message: {request.message}")
    groupchat.reset() # Reset messages for a new request

    try:
        user_proxy.initiate_chat(
            manager,
            message=request.message,
            clear_history=False,
            is_termination_msg=is_termination_msg
        )

        # --- Enhanced Logging ---
        logging.info("--- Chat History ---")
        for i, msg in enumerate(groupchat.messages):
            logging.info(f"Message {i}: {msg}") # Log the potentially incorrect history
        logging.info("--- End Chat History ---")
        # --- End Enhanced Logging ---


        # --- Message Extraction V5 (Workaround for Role Issue) ---
        final_response = "No reply generated." # Default response

        if groupchat.messages:
            # Iterate backwards to find the last message from an AGENT with valid text content
            for msg in reversed(groupchat.messages):
                sender_name = msg.get("name")
                content = msg.get("content")
                content_str = str(content).strip() if content is not None else ""

                # Check if sender is one of the agents and content is valid text (not "None", not escalation)
                if sender_name in [general_agent.name, senior_agent.name] and content_str and content_str != "None" and "I need to escalate" not in content_str:
                    final_response = content_str
                    logging.info(f"Using final valid message from agent {sender_name}: {final_response}")
                    break # Found the last valid agent message

            # If loop finishes and final_response is still the default, log it
            if final_response == "No reply generated.":
                 logging.warning("No suitable text message found from agents in chat history (excluding 'None'/escalation).")

        else: # Handle case where groupchat.messages is empty
            logging.warning("Chat ended with no messages recorded.")
            final_response = "Chat ended with no messages recorded."


        logging.info(f"Sending response: {final_response}")
        return {"response": final_response}

    except Exception as e:
        # ... (keep existing error handling) ...
        logging.error(f"Error during chat processing: {e}", exc_info=True)
        error_message = f"An error occurred: {str(e)}"
        try:
            json.dumps({"response": error_message})
        except TypeError:
            error_message = "An unexpected server error occurred."
        return {"response": error_message}

# --- Run the API server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)