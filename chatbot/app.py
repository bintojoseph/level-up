from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from pinecone import Pinecone
import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

# Initialize FastAPI
app = FastAPI(title="Tech News Chatbot API", description="API for a Tech News Chatbot that answers questions based on latest tech news")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
Context = ""
# Initialize Pinecone
PINECONE_API_KEY = ""  # Replace with your Pinecone API key
PINECONE_INDEX_NAME = ""  # Replace with your Pinecone index name

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="",  # Replace with your Groq API key
    model_name="llama-3.3-70b-versatile"
)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    user_query: str

def get_query_embedding(query):
    """Generate embeddings for the query."""
    return embedding_model.encode(query).tolist()

def search_pinecone(query, top_k=5):
    """Retrieve relevant articles from Pinecone."""
    query_vector = get_query_embedding(query)
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    results = [match["metadata"]["sentence"] for match in response["matches"]]
    return results

def search_web(query):
    """Fetch additional tech news from an external search API (SerpAPI)."""
    SEARCH_API_URL = "https://serpapi.com/search"
    API_KEY = ""  # Replace with your SerpAPI key
    
    params = {"q": query, "api_key": API_KEY}
    response = requests.get(SEARCH_API_URL, params=params)
    
    if response.status_code == 200:
        search_results = response.json().get("results", [])
        return [res["snippet"] for res in search_results][:3]  
    return []

def generate_response(user_query):
    """Generate a response using retrieved knowledge and Groq Llama-3."""
    pinecone_results = search_pinecone(user_query)
    web_results = search_web(user_query)
    
    context = "\n\n".join(pinecone_results + web_results)
    
    prompt_extract = PromptTemplate.from_template(
        """
        This is the combined results from a vector database and web search for a user query about a tech topic.
        Provide a proper response based on the query and result.
        Only reply with a string, no code snippets
        Query: {user_query}
        Context: {context}
        """
    )

    chain_extract = prompt_extract | llm
    res = chain_extract.invoke({"user_query": user_query, "context": context})
    
    return res

@app.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint for chat interaction
    
    Args:
        request: ChatRequest containing knowledge_level and user_query
    
    Returns:
        JSON response with the generated answer
    """
    try:
        response = generate_response(request.user_query)
        
        return {
            "status": "success",
            "response": response.content,
            "query": request.user_query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint to verify the API is running"""
    return {"status": "online", "message": "Tech News Chatbot API is running"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7000, reload=True)
