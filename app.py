from fastapi import FastAPI, HTTPException
from typing import List
from model.similarity_engine import load_wr_data
import numpy as np

app = FastAPI()

# Load player names and similarity matrix at startup
players, similarity_matrix = load_wr_data()

@app.get("/similar", response_model=List[str])
def get_similar_players(player: str, top_k: int = 5):
    if player not in players.values:
        raise HTTPException(status_code=404, detail="Player not found")

    idx = players[players == player].index[0]
    similarities = similarity_matrix[idx]
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]

    return players.iloc[top_indices].tolist()
