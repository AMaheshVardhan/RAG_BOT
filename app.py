import os
import streamlit as st
from mcp_message import MCPMessage
from ingestion_agent import IngestionAgent_MCP
from retrieval_agent import RetrievalAgent_MCP, initialize_vector_store
from llm_response_agent import LLMResponseAgent_MCP
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyDjUXlzJb32blLYPWHZ_32_uO9tpD1uJeo"
genai.configure(api_key=GOOGLE_API_KEY)

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

st.set_page_config(page_title="📄 Agentic RAG Chatbot", layout="wide")
st.title("📄 Agentic RAG Chatbot with MCP")
st.markdown("Ask questions about your uploaded documents!")

coordinator = CoordinatorAgent()

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    st.header("Upload Documents")
    files = st.file_uploader("Upload multiple files", accept_multiple_files=True,
                              type=["pdf", "docx", "pptx", "csv", "txt", "md"])
    if st.button("Process Files"):
        file_infos = []
        for file in files:
            path = f"temp_{file.name}"
            with open(path, "wb") as f:
                f.write(file.read())
            file_infos.append({"name": file.name, "path": path})
        status = coordinator.handle_file_upload(file_infos)
        st.success(f"Files processed. Status: {status}")
        
if st.button("Ask"):
    if query.strip() == "":
        st.warning("Enter a question.")
    elif not coordinator.chat_history:
        st.warning("Please upload and process documents first.")
    else:
        result = coordinator.handle_user_query(query)
        st.markdown(f"**Answer:** {result['answer']}")
        st.markdown("**Source Context:**")
        for c in result["source_context"]:
            st.markdown(f"- **{c['source']}**: {c['text']}")


if coordinator.chat_history:
    st.header("Chat History")
    for item in reversed(coordinator.chat_history):
        st.markdown(f"**Q:** {item['query']}")
        st.markdown(f"**A:** {item['answer']}")
