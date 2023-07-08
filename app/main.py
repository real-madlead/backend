from fastapi import FastAPI
from app.controllers import floor

app = FastAPI()
app.include_router(floor.router, prefix="/api/v1")

@app.get("/api/v1")
def root():
    return {"message": "Hello World"}