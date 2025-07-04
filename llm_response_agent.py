import google.generativeai as genai
from mcp_message import MCPMessage

def format_prompt(query, top_chunks):
    context = "\n\n".join([f"Source: {c['source']}\nText: {c['text']}" for c in top_chunks])
    return f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

def LLMResponseAgent_MCP(message):
    print(f"[LLMResponseAgent] Received: {message}")
    query = message.payload['query']
    top_chunks = message.payload['top_chunks']
    prompt = format_prompt(query, top_chunks)

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    answer = response.text.strip()

    return MCPMessage(
        sender="LLMResponseAgent",
        receiver="CoordinatorAgent",
        type_="FINAL_ANSWER",
        trace_id=message.trace_id,
        payload={
            "answer": answer,
            "source_context": top_chunks
        }
    )