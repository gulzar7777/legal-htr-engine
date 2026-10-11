from fastapi import FastAPI

app = FastAPI(
    title="Legacy Legal Records HTR Engine",
    description="HTR & Archival Document Processing Service",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
