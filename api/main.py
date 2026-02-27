from fastapi import FastAPI

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
