"""FastAPI app exposing recommendation endpoint."""

from fastapi import FastAPI

app = FastAPI()

@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    return {"message": f"Placeholder recommendation for user {user_id}."}
