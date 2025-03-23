nba_game_simulation.py

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from nba_api.stats.endpoints import PlayByPlayV2, LeagueGameLog, WinProbabilityPBP
from nba_api.stats.static import teams

def get_team_id(team_abbr):
    """
    Retrieve the team ID for a given team abbreviation.
    
    Parameters:
        team_abbr (str): Abbreviation for the NBA team (e.g., 'LAL')
        
    Returns:
        int: The team ID.
    """
    nba_teams = teams.get_teams()
    team = next((t for t in nba_teams if t['abbreviation'] == team_abbr), None)
    if team:
        return team['id']
    else:
        raise ValueError("Team abbreviation not found.")

def get_latest_game_id(team_abbr, season='2023-24'):
    """
    Retrieve the latest game ID for a given team and season.
    
    Parameters:
        team_abbr (str): The team abbreviation.
        season (str): The season string, e.g., '2023-24'
        
    Returns:
        str: Game ID if available, otherwise None.
    """
    team_id = get_team_id(team_abbr)
    game_log = LeagueGameLog(team_id_nullable=team_id, season=season).get_data_frames()[0]
    if not game_log.empty:
        # Return the most recent game (assuming the log is sorted by date descending)
        return game_log.iloc[0]['GAME_ID']
    else:
        return None

def get_play_by_play(game_id):
    """
    Fetch the play-by-play data for a specified game.
    
    Parameters:
        game_id (str): The NBA game identifier.
        
    Returns:
        DataFrame: A dataframe containing the play-by-play events.
    """
    pbp = PlayByPlayV2(game_id=game_id)
    df = pbp.get_data_frames()[0]
    # Select only the relevant columns for analysis
    columns_to_keep = ['EVENTNUM', 'EVENTMSGTYPE', 'PCTIMESTRING', 'SCORE', 
                       'HOMEDESCRIPTION', 'VISITORDESCRIPTION']
    return df[columns_to_keep]

def get_win_probability_data(game_id):
    """
    Retrieve and process win probability data for a game.
    
    Parameters:
        game_id (str): The NBA game identifier.
        
    Returns:
        DataFrame: A dataframe with selected win probability columns.
    """
    wp = WinProbabilityPBP(game_id=game_id)
    df = wp.get_data_frames()[0]
    
    # Check for expected columns and select them if available
    expected_columns = ['GAME_ID', 'EVENT_NUM', 'PCTIMESTRING', 'HOME_PTS', 
                        'VISITOR_PTS', 'HOME_PCT', 'VISITOR_PCT']
    available = [col for col in expected_columns if col in df.columns]
    if not available:
        raise ValueError("Expected win probability columns not found.")
    
    return df[available]

def plot_win_probability(wp_df):
    """
    Plot win probability curves for the home and visitor teams over the game.
    
    Parameters:
        wp_df (DataFrame): DataFrame with win probability data.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(wp_df.index, wp_df['HOME_PCT'], label="Home Win Probability")
    plt.plot(wp_df.index, wp_df['VISITOR_PCT'], label="Visitor Win Probability")
    plt.xlabel("Play Number")
    plt.ylabel("Win Probability (%)")
    plt.title("Win Probability Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution for one game (example)
if __name__ == '__main__':
    team_abbr = 'LAL'
    season = '2023-24'
    game_id = get_latest_game_id(team_abbr, season)
    
    if game_id:
        print(f"Fetching data for Game ID: {game_id}")
        play_by_play_df = get_play_by_play(game_id)
        print("Sample Play-by-Play Data:")
        print(play_by_play_df.sample(5))
        
        wp_df = get_win_probability_data(game_id)
        print("Win Probability Data (head):")
        print(wp_df.head())
        plot_win_probability(wp_df)
    else:
        print("No recent games found for the selected team.")

def get_all_games_play_by_play(game_ids):
    """
    Loop through a list of game IDs and compile a DataFrame containing all play-by-play logs.
    
    Parameters:
        game_ids (list): List of NBA game IDs.
        
    Returns:
        DataFrame: Combined DataFrame with play-by-play logs from all games.
    """
    all_games = []
    for gid in game_ids:
        try:
            df = get_play_by_play(gid)
            df['GAME_ID'] = gid  # Add a column to record which game the event came from
            all_games.append(df)
        except Exception as e:
            print(f"Error fetching game {gid}: {e}")
    if all_games:
        return pd.concat(all_games, ignore_index=True)
    else:
        return pd.DataFrame()

# Example usage:
# Suppose you have a list of game IDs (you might retrieve these from LeagueGameLog for each season)
game_ids = ['0022300014', '0022300015', '0022300016']  # Replace with actual game IDs
combined_events_df = get_all_games_play_by_play(game_ids)
print("Combined Play-by-Play Data:")
combined_events_df.head()

import time
import pandas as pd
import requests_cache
from nba_api.stats.endpoints import PlayByPlayV2

# Install cache so repeated requests are faster
requests_cache.install_cache('nba_api_cache', expire_after=86400)

def fetch_play_by_play_with_retry(game_id, max_retries=3, delay=5, timeout=60):
    for attempt in range(max_retries):
        try:
            pbp = PlayByPlayV2(game_id=game_id, timeout=timeout)
            df = pbp.get_data_frames()[0]
            return df
        except Exception as e:
            print(f"Error fetching game {game_id}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds (Attempt {attempt+2}/{max_retries})...")
                time.sleep(delay)
    return None

def generate_play_by_play_data_file(game_ids, filename='play_by_play_data.csv', flush_every=1000):
    """
    Process all game IDs, saving partial results every `flush_every` games.
    
    Parameters:
      - game_ids: list of game IDs
      - filename: output CSV filename
      - flush_every: number of games after which to flush data to CSV
    """
    all_events = []
    for idx, game_id in enumerate(game_ids, start=1):
        print(f"Processing game {game_id} ({idx}/{len(game_ids)})...")
        df = fetch_play_by_play_with_retry(game_id)
        if df is not None:
            df['GAME_ID'] = game_id
            all_events.append(df)
        else: 
            0
            print(f"Skipping game {game_id} after repeated errors.")
        
        # Flush to CSV every flush_every games
        if idx % flush_every == 0:
            combined = pd.concat(all_events, ignore_index=True)
            combined.to_csv(filename, index=False)
            print(f"Flushed data for {idx} games to {filename}")
    
    # Final write after processing all game IDs
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        combined.to_csv(filename, index=False)
        print(f"Final combined play-by-play data saved to {filename}")
    else:
        print("No play-by-play data was retrieved.")

# Example usage:
if __name__ == '__main__':
    # Replace with your list of game IDs
    # game_ids = ['0022000524', '0022000525', '0022000306']  
    generate_play_by_play_data_file(game_ids, filename='play_by_play_data.csv', flush_every=1000)
###############################################################################################
#### START FROM HERE, scraping already completed, will load csv saved on computer 
###############################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from nba_api.stats.endpoints import PlayByPlayV2, LeagueGameLog, WinProbabilityPBP
from nba_api.stats.static import teams

#generate list of all game ids over the last 3 seasons
from nba_api.stats.endpoints import leaguegamefinder
gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00')
# The first DataFrame of those returned is what we want.
games = gamefinder.get_data_frames()[0]
game_ids=games['GAME_ID'].tolist()


play_by_play_df=pd.read_csv('/Users/chriscollins/Desktop/Spring Semester/Simulation/Project/play_by_play_data.csv')

#merge in dates of games
games_merge=games[['GAME_ID','GAME_DATE','SEASON_ID']].copy().astype({'GAME_DATE': 'datetime64[ns]','GAME_ID':'int64'})
play_by_play_df=play_by_play_df.merge(games_merge, on='GAME_ID', how='left')

play_by_play_df['GAME_DATE'].min(), play_by_play_df['GAME_DATE'].max()
#Games range from 1/28/2020 to 3/19/2025

#Investigate the events in the play by play data
play_by_play_df.columns
play_by_play_df['HOMEDESCRIPTION']
# • 1: Made Shot
# • 2: Missed Shot
# • 3: Free Throw
# • 4: Rebound
# • 5: Turnover
# • 6: Foul
# • 7: Violation
# • 8: Substitution
# • 9: Timeout
# • 10: Jump Ball
# • 11: Ejection
# • 12: Start of Period
# • 13: End of Period
# • 18: Instant Replay (I think)

#Fix the score margin - it starts with NA and only shows up for scoring plays
def determine_shot_type_from_row(row):
    """
    Given a row of the dataframe, determine the shot event type with granularity.
    
    Returns:
      - For free throws (EVENTMSGTYPE == 3):
            "SHOT_FT_MISS" if the description indicates a miss,
            otherwise "SHOT_FT_MAKE".
      - For field goal attempts (EVENTMSGTYPE 1 and 2):
            If EVENTMSGTYPE == 1 (made shot): return "SHOT_3PT_MAKE" if a three-point attempt is detected,
              otherwise "SHOT_2PT_MAKE".
            If EVENTMSGTYPE == 2 (missed shot): return "SHOT_3PT_MISS" if a three-point attempt is detected,
              otherwise "SHOT_2PT_MISS".
      - For all other events, return "SHOT_NO_SHOT".
    """
    evt = row['EVENTMSGTYPE']
    
    # Free throws
    if evt == 3:
        desc = row['HOMEDESCRIPTION'] if pd.notna(row['HOMEDESCRIPTION']) else row['VISITORDESCRIPTION']
        if pd.isna(desc):
            return "SHOT_FT_MAKE"  # Default to made free throw if no info is available.
        desc_upper = desc.upper()
        if "MISS" in desc_upper:
            return "SHOT_FT_MISS"
        else:
            return "SHOT_FT_MAKE"
    
    # Field goal attempts (made or missed)
    elif evt in [1, 2]:
        desc = row['HOMEDESCRIPTION'] if pd.notna(row['HOMEDESCRIPTION']) else row['VISITORDESCRIPTION']
        if pd.isna(desc):
            return "SHOT_NO_SHOT"
        desc_upper = desc.upper()
        is_three = ("3PT" in desc_upper or "3-PT" in desc_upper or "3 PTS" in desc_upper)
        if evt == 1:
            return "SHOT_3PT_MAKE" if is_three else "SHOT_2PT_MAKE"
        else:
            return "SHOT_3PT_MISS" if is_three else "SHOT_2PT_MISS"
    
    else:
        return "SHOT_NO_SHOT"

###### Start the process of building the model that will drive simulations
#need to weight every event to a probability for a team
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Helper Functions ---
def parse_time(time_str):
    """
    Convert a time string in the format "MM:SS" to total seconds.
    Returns np.nan if the string cannot be parsed.
    """
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except Exception:
        return np.nan

def parse_margin(margin_str):
    """
    Convert a SCOREMARGIN string (e.g. "+5" or "-3") to an integer.
    """
    try:
        return int(margin_str)
    except:
        return np.nan

# --- Feature Preparation ---

# Assume df is your full play-by-play dataframe (with ~13 million rows) that you’ve saved.
# For modeling possessions, you would ideally aggregate events into possessions.
# For this example, we assume each row is an independent possession outcome.

# Make a copy of your play-by-play dataframe
df_model = play_by_play_df.copy()

# Convert PERIOD, TIME_REMAINING, and SCOREMARGIN
df_model['PERIOD'] = df_model['PERIOD']
df_model['TIME_REMAINING'] = df_model['PCTIMESTRING'].apply(parse_time)
df_model['SCOREMARGIN_NUM'] = df_model['SCOREMARGIN'].apply(parse_margin)

# Create the new SHOT_TYPE feature for each row.
df_model['SHOT_TYPE'] = df_model.apply(determine_shot_type_from_row, axis=1)

# For our initial model, let's include all events, but we add the SHOT_TYPE categorical feature.
# Drop rows with missing key numerical features.
df_model = df_model.dropna(subset=['PERIOD', 'TIME_REMAINING', 'SCOREMARGIN_NUM'])

#rename shot type to type to avoid duplicating "SHOT_" in column names
df_model.rename(columns={'SHOT_TYPE': 'TYPE'}, inplace=True)
# One-hot encode the SHOT_TYPE column.
df_model = pd.get_dummies(df_model, columns=['TYPE'], prefix='SHOT', drop_first=False)
df_model['TYPE'].value_counts()

df_model.columns.to_list()
# Define the predictor features.
# For example, we use PERIOD, TIME_REMAINING, SCOREMARGIN_NUM, and the dummy variables for SHOT_TYPE.
feature_cols = ['PERIOD', 'TIME_REMAINING', 'SCOREMARGIN_NUM'] + [col for col in df_model.columns if col.startswith('SHOT_')]
feature_cols
# The target remains EVENTMSGTYPE (i.e., the outcome for the possession).
target = 'EVENTMSGTYPE'

# Prepare the feature matrix X and target vector y.
X = df_model[feature_cols]
y = df_model[target]

# --- Feature Scaling ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Model Fitting: Multinomial Logistic Regression ---

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_scaled, y)

print("Model training complete.")
print("Model coefficients (each row corresponds to one EVENTMSGTYPE):")
print(model.coef_)

df_model.columns
# --- Simulation Function ---

def simulate_possession(features_dict, scaler, model):
    """
    Given a dictionary of features for a possession, predict the probabilities for each event type,
    and simulate one event outcome.
    
    Parameters:
      - features_dict: dict, e.g. {'PERIOD': 2, 'TIME_REMAINING': 360, 'SCOREMARGIN_NUM': 3}
      - scaler: a fitted StandardScaler
      - model: a trained logistic regression model
      
    Returns:
      - simulated_event: the simulated EVENTMSGTYPE (an integer)
      - probs: array of probabilities corresponding to each event type (model.classes_)
    """
    # Update required feature names to match your one-hot encoding
    required_features = ['PERIOD', 'TIME_REMAINING', 'SCOREMARGIN_NUM', 
                     'SHOT_2PT_MAKE', 'SHOT_3PT_MAKE', 'SHOT_FT_MAKE', 'SHOT_FT_MISS', 'SHOT_NO_SHOT']
    # Set defaults: if no shot type info is provided, assume a non-shot event.
    defaults = {'SHOT_2PT': 0, 'SHOT_3PT': 0, 'SHOT_NO_SHOT': 1}
    for feat in required_features:
        if feat not in features_dict:
            features_dict[feat] = defaults.get(feat, 0)
    
    # Create a DataFrame for a single row of features.
    df_features = pd.DataFrame([features_dict])
    df_features_scaled = scaler.transform(df_features)
    
    # Get predicted probabilities for each event type.
    probs = model.predict_proba(df_features_scaled)[0]
    
    # Simulate an event by sampling according to these probabilities.
    simulated_event = np.random.choice(model.classes_, p=probs)
    return simulated_event, probs
# --- Example Usage of Simulation Function ---
# Example possession: in Period 2, 6 minutes remaining (360 seconds) in the period, score margin +3.
example_features = {
    'PERIOD': 2,
    'TIME_REMAINING': 360,  # seconds
    'SCOREMARGIN_NUM': 3
    # No shot type info provided, so defaults will be used:
    # SHOT_2PT = 0, SHOT_3PT = 0, SHOT_NO_SHOT = 1.
}

event, event_probs = simulate_possession(example_features, scaler, model)
print("Simulated event outcome (EVENTMSGTYPE):", event)
print("Associated probabilities for each event type:", event_probs)

###### Try aggregating events into outcomes
def outcome_to_points(event, features_dict=None):
    """
    Map an EVENTMSGTYPE to a point value.
    
    For this simulation:
      - If EVENTMSGTYPE == 1 (made shot) and features_dict indicates a three-point shot (SHOT_3PT==1), returns 3 points.
      - If EVENTMSGTYPE == 1 (made shot) and it's not a three-pointer, returns 2 points.
      - If EVENTMSGTYPE == 3 (free throw), returns 1 point.
      - All other events yield 0 points.
    
    The features_dict should contain the shot type dummy variables used during training.
    """
    if event == 1:  # Made shot
        if features_dict is not None:
            # Check if the possession is a three-point shot based on the one-hot encoding.
            if features_dict.get('SHOT_3PT', 0) == 1:
                return 3
            else:
                return 2
        return 2  # default if no shot type info provided
    elif event == 3:  # Free throw
        return 1
    else:
        return 0

def simulate_game_from_state(game_state, scaler, model, average_possession_time=24, current_score=None):
    """
    Simulate the remainder of a game given a current game state.
    
    Parameters:
      game_state: dict containing keys:
          - 'PERIOD': current period (should be 4 for fourth quarter)
          - 'TIME_REMAINING': seconds remaining in the current period
          - 'SCOREMARGIN_NUM': current score margin (Team A score - Team B score)
          (Optionally, you might have current absolute scores.)
      scaler: fitted StandardScaler used to scale features.
      model: trained logistic regression model predicting EVENTMSGTYPE.
      average_possession_time: average seconds per possession (default 24).
      current_score: dict (optional) with keys 'score_A' and 'score_B' representing the actual current scores.
    
    Returns:
      final_score_A, final_score_B: simulated final scores for Team A and Team B.
    """
    # Extract current game state values
    period = game_state.get('PERIOD', 4)
    time_remaining = game_state.get('TIME_REMAINING', 0)
    current_margin = game_state.get('SCOREMARGIN_NUM', 0)
    
    # Estimate the number of possessions remaining in the current period.
    total_remaining_possessions = int(time_remaining / average_possession_time)
    # Assume possessions are split evenly between teams.
    possessions_per_team = total_remaining_possessions // 2
    
    # Initialize scores: use actual current scores if provided; otherwise, use margin-splitting.
    if current_score is not None:
        score_A = current_score.get('score_A', current_margin / 2.0)
        score_B = current_score.get('score_B', -current_margin / 2.0)
    else:
        score_A = current_margin / 2.0
        score_B = -current_margin / 2.0
    
    for pos in range(possessions_per_team):
        # Team A's possession:
        features_A = {
            'PERIOD': period,
            'TIME_REMAINING': time_remaining,  # For simplicity, keep fixed; you could update per possession.
            'SCOREMARGIN_NUM': score_A - score_B  # current margin from Team A's perspective.
            # Additional shot type info will be set via simulate_possession defaults if not provided.
        }
        event_A, _ = simulate_possession(features_A, scaler, model)
        points_A = outcome_to_points(event_A, features_A)
        score_A += points_A
        
        # Team B's possession:
        features_B = {
            'PERIOD': period,
            'TIME_REMAINING': time_remaining,
            'SCOREMARGIN_NUM': score_B - score_A
        }
        event_B, _ = simulate_possession(features_B, scaler, model)
        points_B = outcome_to_points(event_B, features_B)
        score_B += points_B

    return score_A, score_B

# Example usage:
# Suppose we have extracted a current game state from a real game in the fourth quarter:
current_game_state = {
    'PERIOD': 4,
    'TIME_REMAINING': 300,   # 5 minutes left in Q4 (300 seconds)
    'SCOREMARGIN_NUM': 4     # Team A is leading by 4 points
}

final_score_A, final_score_B = simulate_game_from_state(current_game_state, scaler, model)
print(f"Simulated final scores: Team A: {final_score_A}, Team B: {final_score_B}")

def monte_carlo_game_simulation(scaler, model, num_simulations=100, game_state=None, average_possession_time=24):
    """
    Run Monte Carlo simulations of a game from a given game state and compute Team A's win percentage.
    
    Parameters:
      scaler: fitted StandardScaler.
      model: trained logistic regression model.
      num_simulations: number of game simulations to run.
      game_state: dict with current game state, e.g. {'PERIOD':4, 'TIME_REMAINING':300, 'SCOREMARGIN_NUM':4}
      average_possession_time: average seconds per possession (default 24).
      
    Returns:
      win_percentage: percentage of simulations where Team A wins.
      results: list of tuples (score_A, score_B) for each simulation.
    """
    if game_state is None:
        # If no game_state provided, set default values (e.g., 4th quarter, 5 minutes left, margin 0)
        game_state = {'PERIOD': 4, 'TIME_REMAINING': 300, 'SCOREMARGIN_NUM': 0}
    
    wins_A = 0
    results = []
    for i in range(num_simulations):
        score_A, score_B = simulate_game_from_state(game_state, scaler, model, average_possession_time)
        results.append((score_A, score_B))
        if score_A > score_B:
            wins_A += 1
    win_percentage = wins_A / num_simulations * 100
    return win_percentage, results

# Example usage:
current_game_state = {
    'PERIOD': 4,
    'TIME_REMAINING': 300,  # 5 minutes left in Q4
    'SCOREMARGIN_NUM': 4    # Team A is leading by 4 points
}

win_pct, game_results = monte_carlo_game_simulation(scaler, model, num_simulations=100, game_state=current_game_state)
print(f"Simulated win percentage for Team A: {win_pct:.1f}%")

# --- Example usage ---
# Make sure that your logistic regression model "model", and the fitted scaler "scaler"
# as well as the function simulate_possession(features_dict, scaler, model) are already defined.
#
# For example, your simulate_possession might look like this:
#
# def simulate_possession(features_dict, scaler, model):
#     df_features = pd.DataFrame([features_dict])
#     df_features_scaled = scaler.transform(df_features)
#     probs = model.predict_proba(df_features_scaled)[0]
#     simulated_event = np.random.choice(model.classes_, p=probs)
#     return simulated_event, probs
#
# Now, run the Monte Carlo simulation:

win_pct, game_results = monte_carlo_game_simulation(scaler, model, num_simulations=100, average_possession_time=18)
print(f"Simulated win percentage for Team A: {win_pct:.1f}%")



##### Create model metrics based off the first model
#first, build a function to get random 4th quarter situations so we can extract GAME_ID
from nba_api.stats.endpoints import WinProbabilityPBP

def evaluate_random_4Q_events(df_model, scaler, model, 
                              simulate_game_from_state_func,
                              n_sims=200, 
                              random_state=42):
    """
    1. Picks one random event in the 4th quarter per game.
    2. Simulates the remainder of each game from that event.
    3. Compares the simulated outcome to the actual final outcome and 
       to the Win Probability from nba_api if available.
    4. Returns a DataFrame with columns:
       [GAME_ID, EVENTNUM, MODEL_WIN_PCT, ACTUAL_OUTCOME, OFFICIAL_WP]
    
    Parameters:
      df_model : DataFrame
          The DataFrame with your events (including PERIOD, TIME_REMAINING, SCOREMARGIN_NUM).
      scaler : StandardScaler
          The fitted scaler used to transform features for your logistic regression model.
      model : LogisticRegression
          The trained logistic regression model predicting EVENTMSGTYPE.
      simulate_game_from_state_func : function
          A function like `simulate_game_from_state(game_state, scaler, model, ...)` 
          that returns (scoreA, scoreB).
      n_sims : int
          Number of Monte Carlo simulations to run for each random event.
      random_state : int
          Random seed for reproducibility.

    Returns:
      A DataFrame with columns: 
        - GAME_ID
        - EVENTNUM
        - MODEL_WIN_PCT : predicted chance that "Team A" wins
        - ACTUAL_OUTCOME : 1 if Team A won, 0 if Team A lost, np.nan if unknown
        - OFFICIAL_WP : Win Probability from nba_api (if found), else np.nan
    """

    np.random.seed(random_state)

    # -------------- Helper Functions --------------

    def pick_random_4th_quarter_events(df):
        """
        Group by GAME_ID, filter for 4th quarter (PERIOD == 4),
        and pick exactly one random row from each group.
        """
        df_4Q = df[df['PERIOD'] == 4].copy()
        # If a game doesn't have 4Q events, it won't appear here.
        
        # We'll define a function to sample one row from each group
        def sample_one(group):
            return group.sample(n=1, random_state=random_state)
        
        random_rows = df_4Q.groupby('GAME_ID', group_keys=False).apply(sample_one)
        return random_rows

    def get_actual_outcome(df, game_id):
        """
        Determine if 'Team A' ended up winning (return 1) or losing (return 0).
        We'll define 'Team A' as the perspective of SCOREMARGIN_NUM > 0 => Team A won.
        
        If your data has absolute final scores, you can do a more direct check.
        """
        game_events = df[df['GAME_ID'] == game_id]
        if game_events.empty:
            return np.nan
        final_row = game_events.iloc[-1]  # last row in chronological order
        final_margin = final_row.get('SCOREMARGIN_NUM', np.nan)
        if pd.isna(final_margin):
            return np.nan
        return 1 if final_margin > 0 else 0

    def get_nba_wp_at_event(game_id, eventnum):
        """
        Fetch WinProbabilityPBP data for a game and return 
        the home team's WP near the given eventnum if possible.
        
        This is approximate; you may need to adapt if your data 
        doesn't match event numbers exactly.
        """
        try:
            wp_data = WinProbabilityPBP(game_id=game_id).get_data_frames()[0]
            # Attempt to match the row with the same EVENT_NUM
            row = wp_data.loc[wp_data['EVENT_NUM'] == eventnum]
            if not row.empty:
                # If 'HOME_PCT' is the column for the home team's WP, we can read it here.
                return row.iloc[-1]['HOME_PCT']  # or 'HOME_POSS_WIN_PCT' in older versions
            return np.nan
        except Exception as e:
            print(f"Error fetching WP data for game {game_id}: {e}")
            return np.nan

    # -------------- Main Logic --------------

    # 1) Pick random 4th quarter events
    random_4Q_events = pick_random_4th_quarter_events(df_model)
    if random_4Q_events.empty:
        print("No 4th-quarter events found in the dataset.")
        return pd.DataFrame(columns=['GAME_ID','EVENTNUM','MODEL_WIN_PCT','ACTUAL_OUTCOME','OFFICIAL_WP'])

    results = []
    
    # 2) For each random event, simulate the remainder, then compare outcomes
    for idx, row in random_4Q_events.iterrows():
        game_id = row['GAME_ID']
        eventnum = row['EVENTNUM']
        
        # Build a game_state dict for your simulation
        game_state = {
            'PERIOD': row.get('PERIOD', 4),
            'TIME_REMAINING': row.get('TIME_REMAINING', 0),
            'SCOREMARGIN_NUM': row.get('SCOREMARGIN_NUM', 0)
            # If you have absolute scores, you'd include them here instead of margin-splitting
        }

        # Run Monte Carlo from this state
        wins = 0
        for _ in range(n_sims):
            scoreA, scoreB = simulate_game_from_state_func(game_state, scaler, model)
            if scoreA > scoreB:
                wins += 1
        model_win_pct = 100.0 * wins / n_sims

        # 3) Compare to actual outcome
        actual_outcome = get_actual_outcome(df_model, game_id)

        # 4) Compare to official WP from nba_api
        official_wp = get_nba_wp_at_event(game_id, eventnum)
        
        # 5) Store in results
        results.append({
            'GAME_ID': game_id,
            'EVENTNUM': eventnum,
            'MODEL_WIN_PCT': model_win_pct,
            'ACTUAL_OUTCOME': actual_outcome,
            'OFFICIAL_WP': official_wp
        })
    
    # Convert to DataFrame
    evaluation_df = pd.DataFrame(results)
    return evaluation_df

evaluation_df = evaluate_random_4Q_events(
    df_model=df_model, 
    scaler=scaler, 
    model=model,
    simulate_game_from_state_func=simulate_game_from_state,  # your function
    n_sims=200,
    random_state=42
)

print(evaluation_df.head(10))