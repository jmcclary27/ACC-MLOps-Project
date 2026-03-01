from fastapi import FastAPI
from pydantic import BaseModel

# Create the FastAPI application instance
app = FastAPI(title="MLOps Model Server")

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running correctly.
    """
    return {
        "status": "ok",
        "message": "API is healthy."
    }

class DocumentRequest(BaseModel):
    text: str

@app.post("/clauses")
def process_clauses(request: DocumentRequest):
    """
    Takes a block of text and returns a mock label and confidence score for each sentence.
    """
    # test sentence split
    sentences = request.text.split(".")
    
    results = []
    for sentence in sentences:
        clean_sentence = sentence.strip()
        if clean_sentence: 
            results.append({
                "sentence": clean_sentence + ".", 
                "label": "acceptable",
                "confidence": 0.91
            })
            
    return results
