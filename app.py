# app.py
import streamlit as st
from mcp_message import MCPMessage
from ingestion_agent import IngestionAgent_MCP
from retrieval_agent import RetrievalAgent_MCP, initialize_vector_store
from llm_response_agent import LLMResponseAgent_MCP
import google.generativeai as genai

# Hardcoded API Key (Only for development!)
genai.configure(api_key="AIzaSyDjUXlzJb32blLYPWHZ_32_uO9tpD1uJeo")

# Initialize vector store
initialize_vector_store()

class CoordinatorAgent:
    def __init__(self):
        self.chat_history = []

    def handle_file_upload(self, files):
        message = MCPMessage(
            sender="UI",
            receiver="IngestionAgent",
            type_="UPLOAD",
            payload={"files": files}
        )
        ingested = IngestionAgent_MCP(message)
        retrieved = RetrievalAgent_MCP(ingested)
        return retrieved.payload["status"]

    def handle_user_query(self, query):
        message = MCPMessage(
            sender="UI",
            receiver="RetrievalAgent",
            type_="QUERY",
            payload={"query": query}
        )
        retrieval_result = RetrievalAgent_MCP(message)
        llm_response = LLMResponseAgent_MCP(retrieval_result)
        self.chat_history.append({
            "query": query,
            "answer": llm_response.payload["answer"],
            "sources": llm_response.payload["source_context"]
        })
        return llm_response.payload

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 Agentic RAG Chatbot")
st.markdown("Upload your documents and ask questions!")

coordinator = CoordinatorAgent()

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    st.header("Upload Documents")
    files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "docx", "pptx", "csv", "txt", "md"])
    if st.button("Process Files"):
        file_infos = []
        for file in files:
            path = f"temp_{file.name}"
            with open(path, "wb") as f:
                f.write(file.read())
            file_infos.append({"name": file.name, "path": path})
        status = coordinator.handle_file_upload(file_infos)
        st.success(f"Files processed. Status: {status}")

st.header("Ask Questions")
query = st.text_input("Your question:")
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        result = coordinator.handle_user_query(query)
        st.markdown(f"**Answer:** {result['answer']}")
        st.markdown("**Sources:**")
        for c in result["source_context"]:
            st.markdown(f"- **{c['source']}**: {c['text']}")

if coordinator.chat_history:
    st.header("Chat History")
    for item in coordinator.chat_history:
        st.markdown(f"**Q:** {item['query']}")
        st.markdown(f"**A:** {item['answer']}")
