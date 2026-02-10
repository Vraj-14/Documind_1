# nothing

import streamlit as st
import requests
import json

# --- 1. LOAD CONFIGURATION ---
# Streamlit automatically parses .streamlit/secrets.toml
try:
    BACKEND_URL = st.secrets["general"]["backend_url"]
except Exception as e:
    st.error(f"Error reading secrets: {e}. Please check [general] section in secrets.toml")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Documind AI", page_icon="📄", layout="wide")
st.title("📄 Documind AI Assistant")

# --- SIDEBAR for Company Selection ---
with st.sidebar:
    st.header("Configuration")
    # Load company names from JSON for the dropdown
    try:
        with open(r'c:\Users\USER\Desktop\Documind\InstaDocs.json', 'r') as f:
            data = json.load(f)
        # Assuming the structure is InstaAPIReport -> Metadata -> CompanyName
        # And assuming for now we have one company per file, but preparing for more.
        company_list = [data.get("InstaAPIReport", {}).get("Metadata", {}).get("CompanyName", "Unknown Company")]
        
        # Filter out any None or empty strings
        company_list = [name for name in company_list if name and name != "Unknown Company"]
        
        company_name = st.selectbox("Select a Company:", options=company_list, key="company_name")
    except FileNotFoundError:
        st.error("InstaDocs.json not found. Please place it in the Documind directory.")
        company_name = st.text_input("Enter Company Name to search within:", key="company_name")
    except Exception as e:
        st.error(f"Error loading company names: {e}")
        company_name = st.text_input("Enter Company Name to search within:", key="company_name")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # For assistant messages, show download and sources
        if message["role"] == "assistant":
            st.download_button(
                label="Download Answer",
                data=message["content"],
                file_name="answer.txt",
                mime="text/plain",
                key=f"download_{len(st.session_state.messages)}_{message['content'][:10]}" # Unique key
            )
            if "sources" in message and message["sources"]:
                with st.expander("View Sources & Citations"):
                    for source in message["sources"]:
                        st.info(f"Source: {source}")

# Chat Input
if prompt := st.chat_input("Ask about your documents..."):

    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                payload = {"question": prompt, "company_name": company_name}
                response = requests.post(f"{BACKEND_URL}/chat", json=payload)

                if response.status_code == 200:
                    response_data = response.json()
                    answer = response_data.get("answer", "Sorry, I couldn't find an answer.")
                    sources = response_data.get("sources", [])

                    st.markdown(answer)
                    st.download_button(
                        label="Download Answer",
                        data=answer,
                        file_name="answer.txt",
                        mime="text/plain"
                    )
                    if sources:
                        with st.expander("View Sources & Citations"):
                            for source in sources:
                                st.info(f"Source: {source}")

                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Backend is not running. Please run 'backend.py'.")