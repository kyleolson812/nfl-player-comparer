from fastapi import FastAPI, HTTPException
from app.similarity_engine import SimilarityEngine

app = FastAPI()
engine = SimilarityEngine()


@app.get("/")
def root():
    return {"message": "NFL Player Comparison API is live!"}


@app.get("/compare/{player_name}")
def compare_players(player_name: str, top_n: int = 5):
    try:
        similar_players = engine.find_similar_players(player_name, top_n=top_n)
        return similar_players.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
