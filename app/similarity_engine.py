import pandas as pd
from nfl_data_py import import_pbp_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    def __init__(self, seasons=[2024]):
        self.seasons = seasons
        self.df = self.load_data()
        self.player_stats = self.build_player_stats()
        self.scaled_stats = self.scale_stats()
        # print(list(self.df.columns))
        # print(self.player_stats)
        # exit()


    def load_data(self) -> pd.DataFrame:
        print(f"Loading play-by-play data for seasons: {self.seasons}")
        df = import_pbp_data(self.seasons)
        print("Available player names:", df["passer_player_name"].unique())
        # Filter for relevant offensive plays
        df = df[df['play_type'].isin(['pass', 'run'])]
        return df

    def build_player_stats(self) -> pd.DataFrame:
        # Simplified offensive stats by player
        grouped = self.df.groupby('passer_player_name').agg({
            'yards_gained': 'sum',
            'pass_attempt': 'sum',
            'complete_pass': 'sum',
            'rush_attempt': 'sum',
            'touchdown': 'sum',
            'interception': 'sum'
        }).dropna()

        # Rename for clarity
        grouped.columns = [
            'total_yards',
            'pass_attempts',
            'completions',
            'rush_attempts',
            'touchdowns',
            'interceptions'
        ]

        return grouped

    def scale_stats(self) -> pd.DataFrame:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.player_stats)
        return pd.DataFrame(scaled, index=self.player_stats.index, columns=self.player_stats.columns)

    def find_similar_players(self, player_name: str, top_n: int = 5):
        if player_name not in self.scaled_stats.index:
            raise ValueError(f"Player '{player_name}' not found in dataset")

        player_vector = self.scaled_stats.loc[[player_name]]
        similarity_scores = cosine_similarity(player_vector, self.scaled_stats)[0]

        similarity_df = pd.DataFrame({
            'passer_player_name': self.scaled_stats.index,
            'similarity': similarity_scores
        }).sort_values(by='similarity', ascending=False)

        # Exclude the player themselves
        return similarity_df[similarity_df['passer_player_name'] != player_name].head(top_n)
        



# Example usage (for debugging / dev only)
if __name__ == "__main__":
    engine = SimilarityEngine()
    print(engine.find_similar_players("D.Montgomery", top_n=5))
