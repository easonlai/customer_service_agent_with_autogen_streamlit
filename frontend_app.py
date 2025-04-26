import streamlit as st
import requests
import json

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000/chat" # URL of your FastAPI backend

# --- Streamlit App UI ---

st.title("Customer Service Agent Chat")
st.write("Ask a question to interact with the customer service agents.")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("What is your question?")

if user_input:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Send request to the backend API
    try:
        response = requests.post(BACKEND_URL, json={"message": user_input})
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Process the response
        backend_response = response.json()
        agent_reply = backend_response.get("response", "Sorry, I didn't get a valid response from the agent.")

        # Add agent response to chat history and display it
        st.session_state.messages.append({"role": "assistant", "content": agent_reply})
        with st.chat_message("assistant"):
            st.markdown(agent_reply)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend API: {e}")
        # Optionally remove the user message if the backend call failed
        # st.session_state.messages.pop()
    except json.JSONDecodeError:
        st.error("Received an invalid response format from the backend.")
        # Optionally remove the user message
        # st.session_state.messages.pop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        # Optionally remove the user message
        # st.session_state.messages.pop()
