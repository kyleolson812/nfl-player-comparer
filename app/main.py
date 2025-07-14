from fastapi import FastAPI, HTTPException, Query
from app.similarity_engine import MultiPositionSimilarityEngine
from typing import Optional

app = FastAPI(title="NFL Multi-Position Player Similarity API", version="2.0.0")

# Initialize the engine
print("Initializing multi-position similarity engine...")
engine = MultiPositionSimilarityEngine()
print("Engine initialized successfully!")

@app.get("/")
def root():
    return {
        "message": "NFL Multi-Position Player Similarity API is live!",
        "available_positions": engine.get_available_positions(),
        "endpoints": {
            "positions": "/positions",
            "players": "/players/{position}",
            "compare": "/compare/{position}/{player_name}",
            "stats": "/stats/{position}/{player_name}"
        }
    }

@app.get("/positions")
def get_positions():
    """Get all available positions"""
    try:
        return {
            "positions": engine.get_available_positions(),
            "description": "Available positions for player comparison"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving positions: {str(e)}")

@app.get("/players/{position}")
def get_players_by_position(position: str):
    """Get all available players for a specific position"""
    try:
        position = position.upper()
        players = engine.list_available_players(position)
        
        if not players:
            raise HTTPException(status_code=404, detail=f"No players found for position: {position}")
        
        return {
            "position": position,
            "count": len(players),
            "players": players
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving players: {str(e)}")

@app.get("/players")
def get_all_players():
    """Get all available players grouped by position"""
    try:
        all_players = engine.list_available_players()
        total_players = sum(len(players) for players in all_players.values())
        
        return {
            "total_players": total_players,
            "by_position": all_players
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving players: {str(e)}")

@app.get("/compare/{position}/{player_name}")
def compare_players(
    position: str, 
    player_name: str, 
    top_n: int = Query(5, ge=1, le=20, description="Number of similar players to return")
):
    """Find similar players within the same position"""
    try:
        position = position.upper()
        similar_players = engine.find_similar_players(player_name, position, top_n=top_n)
        
        return {
            "query_player": player_name,
            "position": position,
            "similar_players": similar_players.to_dict(orient="records")
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats/{position}/{player_name}")
def get_player_stats(position: str, player_name: str):
    """Get raw and scaled stats for a specific player"""
    try:
        position = position.upper()
        stats = engine.get_player_stats(player_name, position)
        
        return {
            "player": player_name,
            "position": position,
            "stats": stats
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/compare-cross-position/{player_name}")
def compare_cross_position(
    player_name: str,
    top_n: int = Query(3, ge=1, le=10, description="Number of similar players per position")
):
    """Compare a player across all positions (experimental)"""
    try:
        results = {}
        positions = engine.get_available_positions()
        
        for position in positions:
            try:
                similar = engine.find_similar_players(player_name, position, top_n=top_n)
                results[position] = similar.to_dict(orient="records")
            except ValueError:
                # Player not found in this position
                continue
        
        if not results:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found in any position")
        
        return {
            "query_player": player_name,
            "similar_by_position": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    positions = engine.get_available_positions()
    total_players = sum(len(engine.list_available_players(pos)) for pos in positions)
    
    return {
        "status": "healthy",
        "positions_loaded": len(positions),
        "total_players": total_players,
        "positions": positions
    }
