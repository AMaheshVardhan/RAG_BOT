import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mcp_message import MCPMessage

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []
        self.metadata = []

    def add(self, embeddings, metadatas):
        self.index.add(np.array(embeddings).astype('float32'))
        self.vectors.extend(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding, top_k=5):
        if self.index.ntotal == 0:
            return []  # Return empty if no vectors are added
    
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = None

def initialize_vector_store(dim=384):
    global vector_store
    vector_store = VectorStore(dim)

def embed_text(texts):
    return embed_model.encode(texts).tolist()

def RetrievalAgent_MCP(message):
    global vector_store
    print(f"[RetrievalAgent] Received: {message}")

    if message.type == "INGESTION_RESULT":
        chunks = message.payload["chunks"]
        texts = [c['text'] for c in chunks]
        embeds = embed_text(texts)
        metadatas = [{"text": c['text'], "source": c['source']} for c in chunks]
        vector_store.add(embeds, metadatas)
        return MCPMessage(
            sender="RetrievalAgent",
            receiver="CoordinatorAgent",
            type_="INGESTION_CONFIRMED",
            trace_id=message.trace_id,
            payload={"status": "success"}
        )

    elif message.type == "QUERY":
        query = message.payload["query"]
        query_emb = embed_model.encode([query])[0]
        top_chunks = vector_store.search(query_emb, top_k=5)
        return MCPMessage(
            sender="RetrievalAgent",
            receiver="LLMResponseAgent",
            type_="RETRIEVAL_RESULT",
            trace_id=message.trace_id,
            payload={"query": query, "top_chunks": top_chunks}
        )
