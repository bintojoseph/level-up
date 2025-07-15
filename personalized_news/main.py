from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import uvicorn

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# Initialize FastAPI app
app = FastAPI(
    title="News Article Transformer API",
    description="API for transforming news articles based on user knowledge level",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key='',  # Replace with your actual key
    model_name="llama-3.3-70b-versatile"
)

# Knowledge level descriptions
knowledge_level_description = {
    -1: "Return the original sentences.",
    0: "Explain these sentences in very simple terms.",
    1: "Rephrase these sentences in a basic way, introducing key concepts.",
    2: "Summarize while keeping the core technical details understandable.",
    3: "Provide a moderately technical explanation.",
    4: "Explain with advanced concepts, assuming prior knowledge.",
    5: "Provide an expert-level, highly detailed explanation.",
}


class ArticleRequest(BaseModel):
    article_content: str = Field(..., description="The content of the news article to transform")
    knowledge_level: Dict[str, int] = Field(
        ..., 
        description="Dictionary mapping tech topics to knowledge levels (0-5)",
        example={
            "AI/ML": 1,
            "Blockchain": 2,
            "Cybersecurity": 3,
            "Cloud Computing": 4,
            "IoT": 5,
            "Web Development": 3,
            "Mobile App Development": 0,
            "Data Science": 2,
            "AR/VR": 4,
            "Quantum Computing": 1
        }
    )
    
class ArticleResponse(BaseModel):
    transformed_article: str

# Function to batch sentences into groups within token limits
def batch_sentences(sentences, max_tokens=300):
    batches, current_batch = [], []
    token_count = 0

    for sentence in sentences:
        estimated_tokens = len(sentence.split())  
        if token_count + estimated_tokens > max_tokens:
            batches.append(" ".join(current_batch))
            current_batch = [sentence]
            token_count = estimated_tokens
        else:
            current_batch.append(sentence)
            token_count += estimated_tokens

    if current_batch:
        batches.append(" ".join(current_batch))

    return batches


def rephrase_batch(sentences, knowledge_level_dict):
    prompt_extract = PromptTemplate.from_template(
            """
            Given below are sentences from a tech news article and the knowledge level of user on different tech topics.
            The description of each level are also given.
            Based on the knowledge level of each user transform the article to make it understandable for the user.
            
            sentences: {sentences}
            
            knowledge_level: {knowledge_level}
            
            knowledge_level_description: {knowledge_level_description}
            
            First identify the tech topics present in the sentences, then transform the article based on the user's knowledge level for those topics.
            If a topic is not present in the knowledge level dictionary, assume a moderate knowledge level (2).
            
            Return the rephrased sentence/sentences
            ### RETURN JSON FORMAT
            content:rephrased sentence/sentences in json format
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
    )

    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={'sentences':sentences,'knowledge_level':knowledge_level_dict,'knowledge_level_description':knowledge_level_description})
    json_parser = JsonOutputParser()
    json_res = json_parser.parse(res.content)
    return str(json_res["content"])


def rephrase_article(article, user_knowledge_level_dict):
    sentences = sent_tokenize(article)
    batches = batch_sentences(sentences)

    rephrased_batches = [rephrase_batch(batch, user_knowledge_level_dict) for batch in batches]

    return " ".join(rephrased_batches)


@app.get("/")
async def root():
    return {"message": "News Article Transformer API"}

@app.post("/transform_article", response_model=ArticleResponse)
async def transform_article(request: ArticleRequest):
    try:
        
        if not request.article_content.strip():
            raise HTTPException(status_code=400, detail="Article content cannot be empty")
            
       
        for topic, level in request.knowledge_level.items():
            if level < -1 or level > 5:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Knowledge level for {topic} must be between 0 and 5"
                )
        
        
        transformed_article = rephrase_article(
            article=request.article_content,
            user_knowledge_level_dict=request.knowledge_level
        )
        
        return ArticleResponse(transformed_article=transformed_article)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing article: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)

