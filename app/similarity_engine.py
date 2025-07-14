import pandas as pd
from nfl_data_py import import_pbp_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MultiPositionSimilarityEngine:
    def __init__(self, seasons=[2024]):
        self.seasons = seasons
        self.df = None
        self.position_stats = {}  # Store stats for each position
        self.position_scalers = {}  # Store scalers for each position
        self.position_scaled_stats = {}  # Store scaled stats for each position
        
        try:
            self.df = self.load_data()
            self.build_all_position_stats()
            print("Successfully initialized multi-position similarity engine")
        except Exception as e:
            print(f"Error initializing engine: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        print(f"Loading play-by-play data for seasons: {self.seasons}")
        try:
            df = import_pbp_data(self.seasons)
            print(f"Raw data shape: {df.shape}")
            
            # Filter for relevant offensive plays
            df = df[df['play_type'].isin(['pass', 'run'])]
            print(f"After filtering for pass/run plays: {df.shape}")
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def build_all_position_stats(self):
        """Build stats for all positions"""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            try:
                print(f"\nBuilding stats for {position}...")
                stats = self.build_position_stats(position)
                if len(stats) > 0:
                    self.position_stats[position] = stats
                    self.position_scaled_stats[position] = self.scale_position_stats(position)
                    print(f"Successfully built stats for {len(stats)} {position}s")
                else:
                    print(f"No sufficient data found for {position}")
            except Exception as e:
                print(f"Error building stats for {position}: {e}")

    def build_position_stats(self, position: str) -> pd.DataFrame:
        """Build stats for a specific position"""
        
        if position == 'QB':
            return self.build_qb_stats()
        elif position == 'RB':
            return self.build_rb_stats()
        elif position == 'WR':
            return self.build_wr_stats()
        elif position == 'TE':
            return self.build_te_stats()
        else:
            raise ValueError(f"Unsupported position: {position}")

    def build_qb_stats(self) -> pd.DataFrame:
        """Build QB-specific stats focusing on passing and rushing"""
        qb_df = self.df[self.df["passer_player_name"].notna()].copy()
        
        if qb_df.empty:
            return pd.DataFrame()
        
        # Aggregate QB stats
        agg_dict = {
            'pass_attempt': 'sum',
            'complete_pass': 'sum', 
            'yards_gained': 'sum',
            'touchdown': 'sum',
            'interception': 'sum',
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum',
            'qb_scramble': 'sum'
        }
        
        # Filter for columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in qb_df.columns}
        
        grouped = qb_df.groupby("passer_player_name").agg(agg_dict).fillna(0)
        
        # Filter for minimum attempts
        min_attempts = 50
        if 'pass_attempt' in grouped.columns:
            grouped = grouped[grouped['pass_attempt'] >= min_attempts]
        
        if grouped.empty:
            return pd.DataFrame()
        
        # Calculate derived metrics
        grouped["completion_pct"] = grouped.get("complete_pass", 0) / grouped.get("pass_attempt", 1).replace(0, 1)
        grouped["yards_per_attempt"] = grouped.get("yards_gained", 0) / grouped.get("pass_attempt", 1).replace(0, 1)
        grouped["td_to_int_ratio"] = grouped.get("touchdown", 0) / grouped.get("interception", 1).replace(0, 1)
        grouped["rush_yards_per_attempt"] = grouped.get("rushing_yards", 0) / grouped.get("rush_attempt", 1).replace(0, 1)
        grouped["rush_td_rate"] = grouped.get("rush_touchdown", 0) / grouped.get("rush_attempt", 1).replace(0, 1)
        
        # Final features for QB comparison
        final_features = [
            "completion_pct",
            "yards_per_attempt",
            "td_to_int_ratio", 
            "rush_yards_per_attempt",
            "rush_td_rate"
        ]
        
        return grouped[final_features].replace([np.inf, -np.inf], 0).fillna(0)

    def build_rb_stats(self) -> pd.DataFrame:
        """Build RB-specific stats focusing on rushing and receiving"""
        rb_df = self.df[self.df["rusher_player_name"].notna()].copy()
        
        if rb_df.empty:
            return pd.DataFrame()
        
        # Aggregate rushing stats
        rush_agg = {
            'rush_attempt': 'sum',
            'rushing_yards': 'sum', 
            'rush_touchdown': 'sum',
            'fumble': 'sum'
        }
        
        rush_agg = {k: v for k, v in rush_agg.items() if k in rb_df.columns}
        rush_stats = rb_df.groupby("rusher_player_name").agg(rush_agg).fillna(0)
        
        # Get receiving stats for RBs
        rec_df = self.df[self.df["receiver_player_name"].notna()].copy()
        if not rec_df.empty:
            rec_agg = {
                'complete_pass': 'sum',  # receptions
                'yards_gained': 'sum',   # receiving yards
                'touchdown': 'sum'       # receiving TDs
            }
            rec_agg = {k: v for k, v in rec_agg.items() if k in rec_df.columns}
            rec_stats = rec_df.groupby("receiver_player_name").agg(rec_agg).fillna(0)
            rec_stats.columns = ['receptions', 'receiving_yards', 'receiving_td']
        else:
            rec_stats = pd.DataFrame()
        
        # Combine rushing and receiving
        combined = rush_stats.copy()
        if not rec_stats.empty:
            combined = combined.join(rec_stats, how='outer').fillna(0)
        else:
            combined['receptions'] = 0
            combined['receiving_yards'] = 0
            combined['receiving_td'] = 0
        
        # Filter for minimum attempts
        min_attempts = 30
        if 'rush_attempt' in combined.columns:
            combined = combined[combined['rush_attempt'] >= min_attempts]
        
        if combined.empty:
            return pd.DataFrame()
        
        # Calculate derived metrics
        combined["rush_yards_per_attempt"] = combined.get("rushing_yards", 0) / combined.get("rush_attempt", 1).replace(0, 1)
        combined["rush_td_rate"] = combined.get("rush_touchdown", 0) / combined.get("rush_attempt", 1).replace(0, 1)
        combined["fumble_rate"] = combined.get("fumble", 0) / combined.get("rush_attempt", 1).replace(0, 1)
        combined["yards_per_reception"] = combined.get("receiving_yards", 0) / combined.get("receptions", 1).replace(0, 1)
        combined["receiving_td_rate"] = combined.get("receiving_td", 0) / combined.get("receptions", 1).replace(0, 1)
        
        # Final features for RB comparison
        final_features = [
            "rush_yards_per_attempt",
            "rush_td_rate",
            "fumble_rate",
            "yards_per_reception", 
            "receiving_td_rate"
        ]
        
        return combined[final_features].replace([np.inf, -np.inf], 0).fillna(0)

    def build_wr_stats(self) -> pd.DataFrame:
        """Build WR-specific stats focusing on receiving"""
        wr_df = self.df[self.df["receiver_player_name"].notna()].copy()
        
        if wr_df.empty:
            return pd.DataFrame()
        
        # Aggregate receiving stats
        agg_dict = {
            'complete_pass': 'sum',  # receptions
            'yards_gained': 'sum',   # receiving yards
            'touchdown': 'sum',      # receiving TDs
            'pass_attempt': 'sum'    # targets
        }
        
        agg_dict = {k: v for k, v in agg_dict.items() if k in wr_df.columns}
        grouped = wr_df.groupby("receiver_player_name").agg(agg_dict).fillna(0)
        
        # Filter for minimum targets
        min_targets = 20
        if 'pass_attempt' in grouped.columns:
            grouped = grouped[grouped['pass_attempt'] >= min_targets]
        
        if grouped.empty:
            return pd.DataFrame()
        
        # Calculate derived metrics
        grouped["catch_rate"] = grouped.get("complete_pass", 0) / grouped.get("pass_attempt", 1).replace(0, 1)
        grouped["yards_per_target"] = grouped.get("yards_gained", 0) / grouped.get("pass_attempt", 1).replace(0, 1)
        grouped["yards_per_reception"] = grouped.get("yards_gained", 0) / grouped.get("complete_pass", 1).replace(0, 1)
        grouped["td_rate"] = grouped.get("touchdown", 0) / grouped.get("pass_attempt", 1).replace(0, 1)
        
        # Final features for WR comparison
        final_features = [
            "catch_rate",
            "yards_per_target",
            "yards_per_reception",
            "td_rate"
        ]
        
        return grouped[final_features].replace([np.inf, -np.inf], 0).fillna(0)

    def build_te_stats(self) -> pd.DataFrame:
        """Build TE-specific stats (similar to WR but may have different thresholds)"""
        # TEs typically have fewer targets than WRs
        te_df = self.df[self.df["receiver_player_name"].notna()].copy()
        
        if te_df.empty:
            return pd.DataFrame()
        
        agg_dict = {
            'complete_pass': 'sum',
            'yards_gained': 'sum',
            'touchdown': 'sum',
            'pass_attempt': 'sum'
        }
        
        agg_dict = {k: v for k, v in agg_dict.items() if k in te_df.columns}
        grouped = te_df.groupby("receiver_player_name").agg(agg_dict).fillna(0)
        
        # Lower threshold for TEs
        min_targets = 15
        if 'pass_attempt' in grouped.columns:
            grouped = grouped[grouped['pass_attempt'] >= min_targets]
        
        if grouped.empty:
            return pd.DataFrame()
        
        # Same metrics as WR
        grouped["catch_rate"] = grouped.get("complete_pass", 0) / grouped.get("pass_attempt", 1).replace(0, 1)
        grouped["yards_per_target"] = grouped.get("yards_gained", 0) / grouped.get("pass_attempt", 1).replace(0, 1)
        grouped["yards_per_reception"] = grouped.get("yards_gained", 0) / grouped.get("complete_pass", 1).replace(0, 1)
        grouped["td_rate"] = grouped.get("touchdown", 0) / grouped.get("pass_attempt", 1).replace(0, 1)
        
        final_features = [
            "catch_rate",
            "yards_per_target", 
            "yards_per_reception",
            "td_rate"
        ]
        
        return grouped[final_features].replace([np.inf, -np.inf], 0).fillna(0)

    def scale_position_stats(self, position: str) -> pd.DataFrame:
        """Scale stats for a specific position"""
        if position not in self.position_stats:
            return pd.DataFrame()
        
        stats = self.position_stats[position]
        if stats.empty:
            return pd.DataFrame()
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(stats)
        self.position_scalers[position] = scaler
        
        return pd.DataFrame(scaled, index=stats.index, columns=stats.columns)

    def find_similar_players(self, player_name: str, position: str, top_n: int = 5):
        """Find similar players within the same position"""
        if position not in self.position_scaled_stats:
            raise ValueError(f"Position '{position}' not available. Available positions: {list(self.position_scaled_stats.keys())}")
        
        scaled_stats = self.position_scaled_stats[position]
        
        if player_name not in scaled_stats.index:
            available_players = list(scaled_stats.index)
            raise ValueError(f"Player '{player_name}' not found in {position} dataset. Available {position}s: {len(available_players)}")

        try:
            player_vector = scaled_stats.loc[[player_name]]
            similarity_scores = cosine_similarity(player_vector, scaled_stats)[0]

            similarity_df = pd.DataFrame({
                'player_name': scaled_stats.index,
                'similarity': similarity_scores,
                'position': position
            }).sort_values(by='similarity', ascending=False)

            # Exclude the player themselves
            result = similarity_df[similarity_df['player_name'] != player_name].head(top_n)
            
            print(f"Found {len(result)} similar {position}s to {player_name}")
            return result
            
        except Exception as e:
            print(f"Error finding similar players: {e}")
            raise

    def get_player_stats(self, player_name: str, position: str):
        """Get raw and scaled stats for a player"""
        if position not in self.position_stats:
            raise ValueError(f"Position '{position}' not available")
            
        if player_name not in self.position_stats[position].index:
            raise ValueError(f"Player '{player_name}' not found in {position} dataset")
            
        return {
            'raw_stats': self.position_stats[position].loc[player_name].to_dict(),
            'scaled_stats': self.position_scaled_stats[position].loc[player_name].to_dict()
        }

    def list_available_players(self, position: str = None):
        """List available players for a position or all positions"""
        if position:
            if position not in self.position_stats:
                return []
            return sorted(list(self.position_stats[position].index))
        else:
            all_players = {}
            for pos in self.position_stats:
                all_players[pos] = sorted(list(self.position_stats[pos].index))
            return all_players

    def get_available_positions(self):
        """Get list of available positions"""
        return list(self.position_stats.keys())


# Example usage
if __name__ == "__main__":
    try:
        engine = MultiPositionSimilarityEngine()
        
        # Show available positions
        positions = engine.get_available_positions()
        print(f"\nAvailable positions: {positions}")
        
        # For each position, show some players
        for position in positions:
            players = engine.list_available_players(position)
            print(f"\n{position} players ({len(players)}): {players[:5]}...")
            
            if players:
                # Test similarity for first player
                test_player = players[0]
                print(f"\nPlayers similar to {test_player} ({position}):")
                similar = engine.find_similar_players(test_player, position, top_n=3)
                print(similar)
                
    except Exception as e:
        print(f"Error: {e}")