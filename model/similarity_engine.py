from nfl_data_py import import_offensive_stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load and preprocess player data
def load_wr_data():
    df = import_offensive_stats([2023])
    wr_df = df[df['position'] == 'WR']
    wr_df = wr_df[['player_name', 'targets', 'receptions', 'receiving_yards', 'receiving_tds']].fillna(0)

    players = wr_df['player_name'].reset_index(drop=True)
    X = wr_df.drop(columns=['player_name'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    similarity_matrix = cosine_similarity(X_scaled)

    return players, similarity_matrix
