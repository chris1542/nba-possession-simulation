nba_game_simulation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from nba_api.stats.endpoints import PlayByPlayV2, LeagueGameLog, WinProbabilityPBP
from nba_api.stats.static import teams
from sklearn.preprocessing import StandardScaler
import time
import scipy.stats as stats
import warnings
from collections import defaultdict

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
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00')
# The first DataFrame of those returned is what we want.
games = gamefinder.get_data_frames()[0]
game_ids=games['GAME_ID'].tolist()


play_by_play_df=pd.read_csv('/Users/collinsch/Downloads/play_by_play_data.csv')

#merge in dates of games
games_merge=games[['GAME_ID','GAME_DATE','SEASON_ID']].copy().astype({'GAME_DATE': 'datetime64[ns]','GAME_ID':'int64'})
play_by_play_df=play_by_play_df.merge(games_merge, on='GAME_ID', how='left')

play_by_play_df['GAME_DATE'].min(), play_by_play_df['GAME_DATE'].max()
#Games range from 1/28/2020 to 3/19/2025

#Investigate the events in the play by play data
play_by_play_df['EVENTMSGTYPE'].unique()
#download a single game of data to inspect
#play_by_play_df[play_by_play_df['GAME_ID']==22401007].to_csv('/Users/chriscollins/Desktop/Spring Semester/Simulation/Project/sample_play_by_play.csv')
#Here are the types of events, as shown in the NBA API github repo 
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

## Drop Duplicates - for some reason the play by play data has every event listed twice
play_by_play_df = play_by_play_df.drop_duplicates()
#it dropped half the values, hopefully will make it faster

#Fix the score margin - it starts with NA and only shows up for scoring plays
#Need to take nonshooting events and classify them as such
#Also need to look at whether it is a miss or a make so we can calculate the score for each team 
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
            return "NO_SHOT"
        desc_upper = desc.upper()
        is_three = ("3PT" in desc_upper or "3-PT" in desc_upper or "3 PTS" in desc_upper)
        if evt == 1:
            return "SHOT_3PT_MAKE" if is_three else "SHOT_2PT_MAKE"
        else:
            return "SHOT_3PT_MISS" if is_three else "SHOT_2PT_MISS"
    
    else:
        return "NO_SHOT"
    
play_by_play_df[play_by_play_df['SCOREMARGIN'].notna()]['SCOREMARGIN']

#Apply this function to every row to classify the event as scoring/non-scoring and for how many points
play_by_play_df['SCOREUPDATEEVENT'] = play_by_play_df.apply(determine_shot_type_from_row, axis=1)

play_by_play_df.head(20)
#looks good


#to speed things up, drop duplicates
## I think each event is listed two times, one from home team perspective one from visiting team perspective
play_by_play_df.shape
play_by_play_df = play_by_play_df.drop_duplicates(subset=['EVENTNUM', 'GAME_ID', 'EVENTMSGTYPE', 'SCOREUPDATEEVENT'], keep='first')
play_by_play_df.shape
play_by_play_df.columns
### EDA - number of events per game
#calculate number of events per game
num_events_per_game=play_by_play_df.groupby('GAME_ID')['EVENTNUM'].max()

plt.figure(figsize=(8, 5))
plt.hist(num_events_per_game, bins=30, edgecolor='black')
plt.xlabel("Number of Events per Game")
plt.ylabel("Number of Games")
plt.title("Number of Events in NBA Games")
plt.xlim(500,900)
plt.show()

#calculate number of shots per game
num_shots_per_game=play_by_play_df[play_by_play_df['EVENTMSGTYPE'].isin([1, 2])].groupby('GAME_ID')['EVENTNUM'].count()
num_shots_per_game.describe()

#calculate number of makes and misses per game for table in report
num_makes_per_game=play_by_play_df[play_by_play_df['EVENTMSGTYPE'].isin([1])].groupby('GAME_ID')['EVENTNUM'].count()
num_misses_per_game=play_by_play_df[play_by_play_df['EVENTMSGTYPE'].isin([2])].groupby('GAME_ID')['EVENTNUM'].count()
num_makes_per_game.describe()
num_misses_per_game.describe()

#### Perform EDA on Shots per minute
## Need to understand how long in between shots to drive how long each possession takes
#it's not perfect because it's not accounting for turnovers, but it's a starting point
#Filter for Field Goal Attempts
fg_df = play_by_play_df[play_by_play_df['EVENTMSGTYPE'].isin([1, 2])]

#Group by Game and Count Attempts
fg_counts = fg_df.groupby('GAME_ID').size()  # counts of FG attempts per game

# Calculate shots per minute assuming a 48-minute game, not perfect, but a start
shots_per_minute = fg_counts / 48.0

# Display summary
print("Field Goal Attempts Per Minute (regulation 48 minutes):")
print(shots_per_minute.describe())

#plot histogram
plt.figure(figsize=(8, 5))
plt.hist(shots_per_minute, bins=30, edgecolor='black')
plt.xlabel("Field Goal Attempts per Minute")
plt.ylabel("Number of Games")
plt.title("Distribution of Field Goal Attempts per Minute")
plt.xlim(2,5)
plt.show()

#look at just makes per minute
makes_df = play_by_play_df[play_by_play_df['EVENTMSGTYPE'].isin([1])]
#Group by Game and Count Attempts
# Calculate makes per minute (assuming a 48-minute game) and print summary statistics
num_makes_per_minute = num_makes_per_game / 48.0
# Create bins with increments of 0.25
bins = np.arange(0, np.ceil(num_makes_per_minute.max()/0.25)*0.25 + 0.25, 0.25)
# Display summary
plt.figure(figsize=(8, 5))
plt.hist(num_makes_per_minute, bins=bins, edgecolor='black')
plt.xlabel("Field Goal Makes per Minute")
plt.ylabel("Number of Games")
plt.title("Distribution of Field Goal Makes per Minute")
plt.xlim(1.0,2.75)
plt.show()



###### Build helper functions that will help to build out the simulation function and open up more EDA
# Helper Function 1: Parse the clock string ("MM:SS") and return total seconds.
# will use to find a random moment in a game and convert time remaining to seconds
def parse_time(time_str): #input = "MM:SS" string, output = int of total seconds
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    except Exception:
        return np.nan

# Helper Function 2: Determine how many possessions are remaining.
# Assumption = 48 minute game, 4 quarters, 12 min quarters 
# Making a huge assumption to fall back on that each possession is roughly 18 seconds. There could be huge variance in this
#This is just a start, will build on this and make specific estimates for duration of time between possessions for each team
# some teams may play with a fast pace, others with a slower pace 
def remaining_possessions(time_remaining, avg_possession_time=18): #input = int of seconds remaining, output = int of possessions remaining
    return int(time_remaining / avg_possession_time)

# Helper Function 3: Calculate score
def outcome_to_points(event, features_dict=None): #input = string of event (ie rebound, shot made, etc), output = int of points scored after considering all possibilities 
    if event == "SHOT_2PT_MAKE":
        return 2
    elif event == "SHOT_3PT_MAKE":
        return 3
    elif event == "SHOT_FT_MAKE":
        return 1
    else:
        return 0

# Helper 4: derive actual outcome from the final event of a game.
# Needed when determining who actually won or lost the game from play_by_play_df
# will use to check if model prediction was correct
def get_actual_outcome_from_game(df, game_id): #input will be play_by_play_df, output will be a 1 or 0, with 1 meaning Team A (home team) won
 # Filter events for the given game and sort chronologically.
    game_events = df[df['GAME_ID'] == game_id].sort_values('EVENTNUM')
    # Select only events with a non-null SCOREMARGIN.
    game_events_with_margin = game_events[game_events['SCOREMARGIN'].notna()]
    
    # If there are events with SCOREMARGIN, use the final one.
    if not game_events_with_margin.empty:
        final_row = game_events_with_margin.iloc[-1]
        try:
            margin = int(final_row['SCOREMARGIN'])
            return 1 if margin > 0 else 0  # Bug fix: explicitly return 0 if margin <= 0.
        except Exception:
            return np.nan
    else:
        # Fallback: try to derive scores from the SCORE column.
        game_events_with_score = game_events[game_events['SCORE'].notna()]
        if game_events_with_score.empty:
            return np.nan
        final_row = game_events_with_score.iloc[-1]
        try:
            parts = final_row['SCORE'].split('-')
            if len(parts) != 2:
                return np.nan
            score_A = int(parts[0].strip())
            score_B = int(parts[1].strip())
            margin = score_A - score_B
            return 1 if margin > 0 else 0
        except Exception:
            return np.nan
        return np.nan

# Helper 5: Parse game scores using the SCORE column
def parse_scores(score_str): #input = SCORE column, output = score for Team A and Team B
    try:
        parts = score_str.split('-')
        if len(parts) != 2:
            return None, None
        score_A = int(parts[0].strip())
        score_B = int(parts[1].strip())
        return score_A, score_B
    except Exception:
        return None, None

#Create a made up game_state to drive the next function
game_state = {
    'PERIOD': 4,
    'TIME_REMAINING': 300,      # 300 seconds remain in Q4 (5 minutes)
    'SCOREMARGIN_NUM': 4.0,       # Team A is leading by 4 points
    'current_event': 'NO_SHOT',   # The most recently observed event state
    'score_A': 50,                # (Optional) Team A's actual current score
    'score_B': 46                 # (Optional) Team B's actual current score
}

#Helper 6: Game State Dynamics to run down the clock based on number of possessions left
def simulate_game_dynamic(game_state, transition_matrix, avg_possession_time=18): 
    #input = state of the game, including period, time remaining in periond, score margin, and last observed event
    #output = the final scores in the simulation, the event sequence (driven by the numbers on nba_api) from the simulation
    # Define time consumption in seconds for each type of event. - this is shaky, there should be variance in time between possessions but this will get us started
    time_consumption = {
        "SHOT_2PT_MAKE": 20,
        "SHOT_3PT_MAKE": 20,
        "SHOT_2PT_MISS": 20,
        "SHOT_3PT_MISS": 20,
        "SHOT_FT_MAKE": 0,   
        "SHOT_FT_MISS": 0,
        "NO_SHOT": 0
    }
    
    period = game_state.get('PERIOD')
    time_remaining = game_state.get('TIME_REMAINING')
    
    # Initialize scores: use given absolute scores if available; if not, use margin-splitting.
    if game_state.get('score_A') is not None and game_state.get('score_B') is not None:
        score_A = game_state['score_A']
        score_B = game_state['score_B']
    else:
        margin = game_state.get('SCOREMARGIN_NUM', 0)
        score_A = margin / 2.0
        score_B = -margin / 2.0
    
    # Starting state: use the provided current event state, defaulting to 'NO_SHOT'
    current_state = game_state.get('current_event')
    
    event_sequence = [] #initialize blank list to hold events from simulation
    
    # Helper: given current_state, sample the next event using the transition matrix.
    def sample_next_event(current_state):
        if current_state not in transition_matrix.index:
            return "NO_SHOT"
        probs = transition_matrix.loc[current_state].values
        possible_states = transition_matrix.columns.tolist()
        return np.random.choice(possible_states, p=probs)
    
    # Simulate until time runs out.
    while time_remaining > 0:
        next_event = sample_next_event(current_state)
        event_sequence.append(next_event)
        
        # Update the score based on the outcome. 
        # There's an assumption that Team A and Team B will alternate possessions, not perfect but it will work
        points = outcome_to_points(next_event)
        if len(event_sequence) % 2 == 1:
            score_A += points
        else:
            score_B += points
        
        # Determine length of time for event, subtract it from TIME_REMAINING.
        consumed = time_consumption.get(next_event, avg_possession_time)
        time_remaining = max(time_remaining - consumed, 0)
        
        # Update the current state for the next iteration.
        current_state = next_event
    
    return (score_A, score_B), event_sequence



# MODEL 1

######## Build Transition Matrix using Markov Chains
def build_transition_matrix(df, state_col='SCOREUPDATEEVENT'): #input = play_by_play_df, output = transition matrix that will show probabilities of transitioning from one event to the next
        #pandas crosstab function will be important here 
    # Sort by game and event number.
    df_sorted = df.sort_values(['GAME_ID', 'EVENTNUM'])
    
    # For each game, shift the state column to obtain the next state.
    df_sorted['NEXT_STATE'] = df_sorted.groupby('GAME_ID')[state_col].shift(-1)
    
    # Drop rows where NEXT_STATE is NA, hopefully that's just the last event in the game
    df_transitions = df_sorted.dropna(subset=['NEXT_STATE'])
    
    # Count transitions in a contingency table.
    transition_counts = pd.crosstab(df_transitions[state_col], df_transitions['NEXT_STATE'])
    
    # Convert counts to probabilities by dividing each row by its row sum.
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    
    return transition_matrix

build_transition_matrix(play_by_play_df.iloc[:, :600], state_col='SCOREUPDATEEVENT') # test on first 600 events, it works

####### MARKOV CHAINS
# Function to run Monte Carlo simulation for one game state using the Markov-chain simulation:
#use the transition matrix built above
def monte_carlo_game_simulation_markov(game_state, transition_matrix, n_simulations=200, avg_possession_time=16):
    #inputs = game_state dict, transition matrix, number of simulations to run, and average possession time
    #output = winning percentage for Team A    
    team_A_wins = 0
    for _ in range(n_simulations):
        (score_A, score_B), _ = simulate_game_dynamic(game_state, transition_matrix, avg_possession_time)
        if score_A > score_B:
            team_A_wins += 1
    win_pct_A = 100.0 * team_A_wins / n_simulations
    return {'win_pct_A': win_pct_A}

# Main function to evaluate multiple games:
def evaluate_multiple_games(df, transition_matrix, n_games=1000, n_simulations=500, avg_possession_time=16, random_state=42):
    #input = play_by_play_df, transition matrix, number of games to simulate, number of simulations to run, average possession time, and seed for replication
    #output = df with game_id, eventnumber, model win percentage, actual outcome, and whether or not the model is correct
    
    #set a random seed 
    np.random.seed(random_state)
    #initialize a blank list to later append
    results = []
    
    # Sample n_games unique GAME_ID values from play_by_play_df.
    unique_game_ids = df['GAME_ID'].unique()
    if len(unique_game_ids) < n_games:
        n_games = len(unique_game_ids)
    #sample game_ids without replacement
    sample_game_ids = np.random.choice(unique_game_ids, size=n_games, replace=False)
    
    #loop through each game ID and calculate the time remaining in seconds in the game
    for game_id in sample_game_ids:
        game_df = df[df['GAME_ID'] == game_id].copy()
        if 'TIME_REMAINING' not in game_df.columns:
            game_df['TIME_REMAINING'] = game_df['PCTIMESTRING'].apply(parse_time)
        
        # Focus on 4th quarter events with nonzero time remaining (> 30 sec)
        game_q4 = game_df[(game_df['PERIOD'] == 4) & (game_df['TIME_REMAINING'] > 30)]
        if game_q4.empty:
            continue
        event_row = game_q4.sample(n=1, random_state=random_state).iloc[0]
        
        #build the game state dictionary 
        game_state = {
            'PERIOD': int(event_row['PERIOD']),
            'TIME_REMAINING': int(parse_time(event_row['PCTIMESTRING'])),
            'SCOREMARGIN_NUM': float(event_row.get('SCOREMARGIN_NUM', 0)),
            'current_event': event_row.get('SCOREUPDATEEVENT', 'NO_SHOT'),
            'score_A': None,
            'score_B': None
        }
        
        #simluate the game using the function from above
        simulation_result = monte_carlo_game_simulation_markov(game_state, transition_matrix, n_simulations=n_simulations, avg_possession_time=avg_possession_time)
        #calculate wining percentage for Team A, becomes a proxy for probability of winning
        model_win_pct = simulation_result.get('win_pct_A', np.nan)
        #pull the actual outcome from the game to compare winning percentage to actual outcome
        actual_outcome = get_actual_outcome_from_game(game_df, game_id)
        
        # Determine if the prediction was correct.
        if pd.isna(actual_outcome):
            prediction_status = "UNKNOWN"
             #50% defaults to a Team A prediction
        elif (model_win_pct >= 50 and actual_outcome == 1) or (model_win_pct < 50 and actual_outcome == 0):
            prediction_status = "CORRECT"
        else:
            prediction_status = "INCORRECT"
        #append to the list initialized above, later converted to dataframe for ease of use     
        results.append({
            'GAME_ID': game_id,
            'RANDOM_EVENTNUM': event_row['EVENTNUM'],
            'MODEL_WIN_PCT': model_win_pct,
            'ACTUAL_OUTCOME': actual_outcome,
            'PREDICTION': prediction_status
        })
    
    return pd.DataFrame(results)

### Test the model! Building a transition matrix and evaluating 1000 games 
#build transition matrix
transition_matrix = build_transition_matrix(play_by_play_df, state_col='SCOREUPDATEEVENT')

# Run evaluation for 1000 games, 105 simulations
evaluation_df = evaluate_multiple_games(play_by_play_df, transition_matrix, n_games=200, n_simulations=100, avg_possession_time=22, random_state=36)
print("Evaluation Results:")
print(evaluation_df.head(10))

evaluation_df['PREDICTION'].value_counts()
#model correctly predicts 531 of 1000 games. Not great, could be random luck
# ran it many times again, 53% was the best it could do. yikes 
#this model is lacking specificity in the possession time between teams

# Use the first model’s functions to run many simulations 
# and record the possession durations (i.e. time consumed per shot event)
# Note: In simulate_game_dynamic, the time consumption for each event type is fixed.
# We use a subset of shot events with nonzero duration.

# Build the transition matrix using the first model function
transition_matrix = build_transition_matrix(play_by_play_df, state_col='SCOREUPDATEEVENT')

# Define an initial game state for simulation (example values)
game_state = {
    'PERIOD': 4,
    'TIME_REMAINING': 300,      # 300 seconds remain in Q4 (5 minutes)
    'SCOREMARGIN_NUM': 4.0,       # Team A is leading by 4 points
    'current_event': 'NO_SHOT',   # Last observed event
    'score_A': 50,              # Team A’s current score
    'score_B': 46               # Team B’s current score
}

# Define the time consumption mapping used in simulate_game_dynamic()
time_consumption = {
    "SHOT_2PT_MAKE": 16,
    "SHOT_3PT_MAKE": 16,
    "SHOT_2PT_MISS": 16,
    "SHOT_3PT_MISS": 16,
    "SHOT_FT_MAKE": 0,   
    "SHOT_FT_MISS": 0,
    "NO_SHOT": 0
}

# We focus on field goal attempts (i.e. shot events with nonzero duration)
shot_event_types = {"SHOT_2PT_MAKE", "SHOT_3PT_MAKE", "SHOT_2PT_MISS", "SHOT_3PT_MISS"}

# Run many simulations using simulate_game_dynamic (first model)
n_simulations = 500
home_shot_durations = []
visitor_shot_durations = []

for _ in range(n_simulations):
    # simulate_game_dynamic returns (final_scores, event_sequence)
    # The event_sequence is a list of events (strings)
    _, events = simulate_game_dynamic(game_state, transition_matrix, avg_possession_time=18)
    # In the first model, possessions alternate:
    # odd-indexed possessions (0, 2, 4, …) are assigned to Team A (home)
    # even-indexed possessions (1, 3, 5, …) are assigned to Team B (visitor)
    for idx, event in enumerate(events):
        if event in shot_event_types:
            if idx % 2 == 0:  # Team A (home)
                home_shot_durations.append(time_consumption.get(event, 18))
            else:           # Team B (visitor)
                visitor_shot_durations.append(time_consumption.get(event, 18))

# Convert durations into DataFrames for analysis
import matplotlib.pyplot as plt

home_df = pd.DataFrame({'duration': home_shot_durations, 'team': 'home'})
visitor_df = pd.DataFrame({'duration': visitor_shot_durations, 'team': 'visitor'})
df_durations = pd.concat([home_df, visitor_df], ignore_index=True)

# Print summary statistics
print("Home Team Shot Duration Summary:")
print(home_df['duration'].describe())
print("\nVisitor Team Shot Duration Summary:")
print(visitor_df['duration'].describe())

# Plot histograms to examine the distribution of shot durations
plt.figure(figsize=(12, 6))
plt.hist(home_df['duration'], bins=30, alpha=0.6, label='Home Team')
plt.hist(visitor_df['duration'], bins=30, alpha=0.6, label='Visitor Team')
plt.xlabel("Time Between Shots (seconds)")
plt.ylabel("Frequency")
plt.title("Distribution of Time Between Shots in Simulated Games (Model 1)")
plt.legend()
plt.show()

# Create side-by-side boxplots for comparison
plt.figure(figsize=(8, 6))
df_durations.boxplot(column='duration', by='team')
plt.xlabel("Team")
plt.ylabel("Time Between Shots (seconds)")
plt.title("Boxplot of Time Between Shots by Team (Model 1)")
plt.suptitle("")
plt.show()

##### Model 2 
## The first model did not perform that well and made some leaps in logic (all teams have same shooting percentage, all teams play at same pace)
## Changes - create a transition matrix specific to the team(s) being simulated, with 50% weighted for last 5 games, 
## Weight transition percentages by recent games - 50% for last 5 games, 25% for games 6-10, and 25% for games 11-25
def build_weighted_team_transition_matrix(df, team_id, state_col='SCOREUPDATEEVENT', simulation_date=None):
    #input - play_by_play_df which will be filtered to a team and the last 25 games they played based on their team_id
    #output - transition matrix for team in question
    # Filter dataframe for games that involve this team.
    team_df = df[(df['PLAYER1_TEAM_ID'] == team_id) | (df['PLAYER2_TEAM_ID'] == team_id)]
    
    # Convert GAME_DATE to datetime format
    team_df['GAME_DATE'] = pd.to_datetime(team_df['GAME_DATE'])
    
    # Filter to only games that occurred before the simulation date.
    simulation_date = pd.to_datetime(simulation_date)
    team_df = team_df[team_df['GAME_DATE'] < simulation_date]

    # Get unique games sorted by date in ascending order. This sets us up to grab the last 25 games
    team_games = team_df[['GAME_ID', 'GAME_DATE']].drop_duplicates().sort_values('GAME_DATE', ascending=True)

    # Grab ids from the last 25 games the team played
    recent_game_ids = list(team_games['GAME_ID'].tail(25))
    
    # Split into three groups: last 5 games, previous 5 games, and previous 15 games.
    if len(recent_game_ids) < 25:
        print(f"Warning: Only {len(recent_game_ids)} games available before {team_df['GAME_DATE']}. All available games will be weighted equally.")
        group1 = recent_game_ids  # Use all games in one group for equal weighting.
        group2 = []
        group3 = []
    else:
        group1 = recent_game_ids[-5:]       # Last 5 games.
        group2 = recent_game_ids[-10:-5]      # Games 6 through 10.
        group3 = recent_game_ids[:-10]        # Games 11 through 25.
    def transition_matrix_for_games(game_ids):
        sub_df = df[df['GAME_ID'].isin(game_ids)].sort_values(['GAME_ID', 'EVENTNUM']).copy()
        sub_df['NEXT_STATE'] = sub_df.groupby('GAME_ID')[state_col].shift(-1)
        sub_df = sub_df.dropna(subset=['NEXT_STATE'])
        counts = pd.crosstab(sub_df[state_col], sub_df['NEXT_STATE'])
        prob_matrix = counts.div(counts.sum(axis=1), axis=0)
        return prob_matrix
    
    tm1 = transition_matrix_for_games(group1)
    tm2 = transition_matrix_for_games(group2)
    tm3 = transition_matrix_for_games(group3) if len(group3) > 0 else None
    
    # Combine the indices and columns from all matrices.
    all_states = set(tm1.index).union(set(tm1.columns))
    if tm2 is not None:
        all_states = all_states.union(set(tm2.index)).union(set(tm2.columns))
    if tm3 is not None:
        all_states = all_states.union(set(tm3.index)).union(set(tm3.columns))
    all_states = sorted(all_states)
    
    # Reindex each matrix and fill missing probabilities with 0.
    tm1 = tm1.reindex(index=all_states, columns=all_states, fill_value=0)
    tm2 = tm2.reindex(index=all_states, columns=all_states, fill_value=0) if tm2 is not None else 0
    tm3 = tm3.reindex(index=all_states, columns=all_states, fill_value=0) if tm3 is not None else 0
    
    # Combine matrices using the desired weights.
    if isinstance(tm3, int):  # i.e., tm3 doesn't exist
        combined = 0.5 * tm1 + 0.5 * tm2
    else:
        combined = 0.5 * tm1 + 0.25 * tm2 + 0.25 * tm3

    return combined

# Example usage:
# Suppose you want to build a weighted transition matrix for Lakers on 2023-02-21.
team_id_example = get_team_id('ATL')
simulation_date = '2023-02-21'  # or datetime string that can be parsed
weighted_transition_matrix = build_weighted_team_transition_matrix(play_by_play_df, team_id_example, state_col='SCOREUPDATEEVENT', simulation_date=simulation_date)
weighted_transition_matrix

### get weighted possessions per team
## Filter to last 25 games for each team, weighting the same as above, to calculate the average time between shots
def get_team_possession_times(df, team_id, simulation_date):
    #input - play_by_play_df and team_id
    #output = array of time between shots for a team in seconds 
    simulation_date = pd.to_datetime(simulation_date)

    # Filter for field goal attempts.
    fg_df = df[df['EVENTMSGTYPE'].isin([1, 2])].copy()
    
    # Filter for events involving the team. Adjust the team columns as necessary.
    team_fg = fg_df.loc[(fg_df['PLAYER1_TEAM_ID'] == team_id) | (fg_df['PLAYER2_TEAM_ID'] == team_id)].copy()
    
    # Use .loc to assign a new column without a SettingWithCopyWarning.
    team_fg.loc[:, 'TIME_REMAINING'] = team_fg['PCTIMESTRING'].apply(parse_time)
    
    # Sort by GAME_ID and EVENTNUM so that events are in order.
    team_fg = team_fg.sort_values(['GAME_ID', 'EVENTNUM'])

    team_fg = team_fg[team_fg['GAME_DATE'] < simulation_date]
    
    possession_times = []
    # Group by GAME_ID (i.e. each game) and compute the time differences.
    for game_id, group in team_fg.groupby('GAME_ID'):
        times = group['TIME_REMAINING'].values
        if len(times) > 1:
            # Calculate differences between consecutive times.
            diffs = times[:-1] - times[1:]
            # Only include differences that are positive (clock runs down)
            valid_diffs = diffs[(diffs > 0)]
            possession_times.extend(valid_diffs)
    return np.array(possession_times)

len(get_team_possession_times(play_by_play_df, team_id_example,simulation_date=simulation_date))
#looks good

#function to weight the possession times
def get_weighted_team_possession_stats(df, team_id, simulation_date):
    #input= df from get_possession_stats, a team_id, and a simulation date
    # output = weighted average and standard dev of possession times over last 25 games
    
    df = df.copy()  # Work on a copy to avoid modifying the original data.
    df.loc[:, 'GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Get the unique game IDs for the team.
    team_games = df.loc[((df['PLAYER1_TEAM_ID'] == team_id) | (df['PLAYER2_TEAM_ID'] == team_id)),
                        ['GAME_ID', 'GAME_DATE']].drop_duplicates().sort_values('GAME_DATE', ascending=True)
    
    simulation_date = pd.to_datetime(simulation_date)
    
    # Only games before simulation_date.
    team_games = team_games[team_games['GAME_DATE'] < simulation_date]
    
    # Get the 25 most recent games BEFORE the simulation date.
    recent_game_ids = team_games['GAME_ID'].tail(25).tolist()
    
    # Sorting these in chronological order ensures group1 (last 5 games) are the most recent.
    recent_game_ids = sorted(recent_game_ids)
    
    # Split into groups.
    group1 = recent_game_ids[-5:]  # Most recent 5 games.
    group2 = recent_game_ids[-10:-5]  # Games 6-10.
    group3 = recent_game_ids[:-10]  # Games 11-25.
    
    def get_group_stats(game_ids):
        sub_df = df[df['GAME_ID'].isin(game_ids)]
        poss_times = get_team_possession_times(sub_df, team_id, simulation_date)
        if len(poss_times) == 0:
            return None, None
        return np.mean(poss_times), np.std(poss_times)
    
    mean1, std1 = get_group_stats(group1)
    mean2, std2 = get_group_stats(group2)
    mean3, std3 = get_group_stats(group3) if len(group3) > 0 else (None, None)
    
    groups = []
    weights = []
    for (m, s, w) in [(mean1, std1, 0.5), (mean2, std2, 0.25), (mean3, std3, 0.25)]:
        if m is not None:
            groups.append((m, s, w))
    if not groups:
        return 18, 3  # Fallback to the defaults we used for the first model
    
    weighted_mean = np.average([m for m, s, w in groups], weights=[w for m, s, w in groups])
    weighted_std = np.average([s for m, s, w in groups], weights=[w for m, s, w in groups])
    return weighted_mean, weighted_std
# Example usage:
team_id_example = get_team_id('BOS')  #ie Celtics.
simulation_date = '2023-02-21'
weighted_mean, weighted_std = get_weighted_team_possession_stats(play_by_play_df, team_id_example, simulation_date=simulation_date)
print(f"Weighted possession time stats for team {team_id_example} prior to {simulation_date}:")
print("Mean:", weighted_mean, "Std:", weighted_std)

## Finally, we are ready to simulate using 
# 1. A Transition Matrix built for each individual team
# 2. A weighted possession time for each team
play_by_play_df.head()
def simulate_game_dynamic_team(game_state, home_stats, visitor_stats, avg_possession_time_fallback=18):
    #input - game_state dictionary with keys of period, time remaining in the period, score_A, and score_B, and the last event
    # input - home_stats dictionary (transition matrix and possession time for home team)
    # input - visitor_stats dictionary (transition matrix and possession time for visitor team)
    # input - avg_possession_time_fallback (default = 18 seconds)
    #output - final score for each team and an event sequence

    #calculate time remaining in the game
    period = game_state.get('PERIOD', 4)
    time_remaining = game_state.get('TIME_REMAINING', 0)
    current_state = game_state.get('current_event', 'NO_SHOT')
    
    #set scores for the game, ran into errors so had to check for data type and implement a backup option 
    if (isinstance(game_state.get('score_A'), (int, float)) and 
        isinstance(game_state.get('score_B'), (int, float))):
        score_A = game_state['score_A']
        score_B = game_state['score_B']
    else: 
        margin = game_state.get('SCOREMARGIN_NUM', 0)
        if pd.isna(margin):
            margin = 0
        score_A = margin / 2.0
        score_B = -margin / 2.0
    
    #initialize empty list to store the future event sequence 
    event_sequence = []
    
    #give current state, sample next event using transition matrix
    def sample_next_event(transition_matrix, current_state):
        if current_state not in transition_matrix.index:
            return "NO_SHOT"
        probs = transition_matrix.loc[current_state].values.astype(float)
        total=probs.sum()
        # If the row sums to zero, we can’t sample. Return a default event.
        if total == 0:
            return "NO_SHOT"
        norm_probs = probs / total
        possible_states = transition_matrix.columns.tolist()
        return np.random.choice(possible_states, p=norm_probs)


    def sample_possession_time(mean, std):
        sample=np.random.normal(mean, std)
        return max(sample, 5)
    
    #shot events that run time off the clock
    shot_events = {"SHOT_2PT_MAKE", "SHOT_3PT_MAKE", "SHOT_2PT_MISS", "SHOT_3PT_MISS"}

    #run the clock out while simulating each subsequent event for the home team
    while time_remaining>0: 
        next_event_home=sample_next_event(home_stats['transition_matrix'], current_state)
        event_sequence.append(("home", next_event_home))
        points_home = outcome_to_points(next_event_home)
        score_A += points_home
    
        if next_event_home in shot_events:
            #run clock pulling from normal distribution of home team possession times
            consumed = sample_possession_time(home_stats['mean'], home_stats['std'])
        else:
            consumed = avg_possession_time_fallback
        
        time_remaining = max(time_remaining - consumed, 0)
        current_state = next_event_home  # update the next current_state

        if time_remaining <= 0:
            break

    # simulate visitor possessions
    next_event_visitor = sample_next_event(visitor_stats['transition_matrix'], current_state)
    event_sequence.append(("visitor", next_event_visitor))
    points_visitor= outcome_to_points(next_event_visitor)
    score_B += points_visitor
    if next_event_visitor in shot_events:
        #run clock pulling from normal distribution of visitor team possession times
        consumed = sample_possession_time(visitor_stats['mean'], visitor_stats['std'])
    else:
        consumed = avg_possession_time_fallback
    
    time_remaining = max(time_remaining - consumed, 0)
    current_state = next_event_visitor  # update the next current_state for the visiting team

    return (score_A, score_B), event_sequence

#Example usage: 
# Suppose you want to simulate a game with the following game state:
game_state = {
    'PERIOD': 4,
    'TIME_REMAINING': 300,      # 300 seconds remain in Q4 (5 minutes)
    'SCOREMARGIN_NUM': 4.0,       # Team A is leading by 4 points
    'current_event': 'NO_SHOT',   # The most recently observed event state
    'score_A': 50,                # (Optional) Team A's actual current score
    'score_B': 46                 # (Optional) Team B's actual current score
}
# Get transition matrices and possession stats for both teams.
home_team_id = get_team_id('LAL')
visitor_team_id = get_team_id('BOS')
home_tm = build_weighted_team_transition_matrix(play_by_play_df, home_team_id, state_col='SCOREUPDATEEVENT', simulation_date=simulation_date)
visitor_tm = build_weighted_team_transition_matrix(play_by_play_df, visitor_team_id, state_col='SCOREUPDATEEVENT', simulation_date=simulation_date)
home_mean, home_std = get_weighted_team_possession_stats(play_by_play_df, home_team_id, simulation_date=simulation_date)
visitor_mean, visitor_std = get_weighted_team_possession_stats(play_by_play_df, visitor_team_id, simulation_date=simulation_date)
home_stats = {
    'transition_matrix': home_tm,
    'mean': home_mean,
    'std': home_std
}
visitor_stats = {
    'transition_matrix': visitor_tm,
    'mean': visitor_mean,
    'std': visitor_std
}
# Simulate the made up game. - this takes a little while, circle back to examine again to see if there's anything that can be done to speed it up
final_score, event_sequence = simulate_game_dynamic_team(game_state, home_stats, visitor_stats, avg_possession_time_fallback=18)


play_by_play_df[play_by_play_df['EVENTMSGTYPE']==10]
#event type 10 is a jump ball, look for the first instance of a 10 for a game id to split between home and visitor teams

### Test the simulation function across many teams and across many games
def evaluate_multiple_games_generic(df, n_games=100, n_simulations=200, avg_possession_time_fallback=18, random_state=42, use_fourth_quarter=False):
    # input - play_by_play_df, number of games to simulate, number of simulations to run for each game, 
    # input - a fallback for average possession time, and a random state for replication
    # output - df with game_id, game date, event number the simulation started from, percentage of time Team A wins from the simulation
    # output - a 1 or 0 to represent win/loss in the actual game, whether or not prediction was correct
    # output - mean and std dev of possession time for the team

    np.random.seed(random_state)
    results = []

    # get n_games worth of unique game_ids from the play_by_play_df
    unique_game_ids = df['GAME_ID'].unique()
    if len(unique_game_ids) < n_games:
        n_games = len(unique_game_ids)
    sample_game_ids = np.random.choice(unique_game_ids, size=n_games, replace=False)

    # loop through game ids to run simulations
    for idx, game_id in enumerate(sample_game_ids, start=1):
        print(f"Simulating game {idx}/{n_games}...")
        game_df = df[df['GAME_ID'] == game_id].copy()
        # Make sure TIME_REMAINING exists.
        if 'TIME_REMAINING' not in game_df.columns:
            game_df['TIME_REMAINING'] = game_df['PCTIMESTRING'].apply(parse_time)
        # find the date the game was played to build average possession time and transition matrices based off this date
        # if for some reason it can't pick up the game date, default to today to build the possession time and transition matrix
        try:
            game_date = pd.to_datetime(game_df['GAME_DATE'].iloc[0])
        except Exception:
            game_date = pd.Timestamp('today')

        # build helper function to find home and visitor team IDs to build possession stats and transition matrices
        def determine_home_and_visitor_team_ids(game_df):
            # input = dataframe of an individual game
            # output = home and visitor team IDs

            jump_ball_events = game_df[game_df['EVENTMSGTYPE'] == 10]
            if not jump_ball_events.empty:
                first_jump = jump_ball_events.iloc[0]
                home_team_id = first_jump['PLAYER1_TEAM_ID']
                visitor_team_id = first_jump['PLAYER2_TEAM_ID']
            return home_team_id, visitor_team_id
        # use the helper function built above
        # Identify the home team ID from the first event (excluding tip-offs) with a non-null HOMEDESCRIPTION.
        home_team_id, visitor_team_id = determine_home_and_visitor_team_ids(game_df)

        # Build team-specific weighted transition matrices and compute possession stats for both teams.
        home_tm = build_weighted_team_transition_matrix(df, home_team_id, state_col='SCOREUPDATEEVENT', simulation_date=game_date)
        visitor_tm = build_weighted_team_transition_matrix(df, visitor_team_id, state_col='SCOREUPDATEEVENT', simulation_date=game_date)
        home_mean, home_std = get_weighted_team_possession_stats(df, home_team_id, simulation_date=game_date)
        visitor_mean, visitor_std = get_weighted_team_possession_stats(df, visitor_team_id, simulation_date=game_date)
        home_stats = {
            'transition_matrix': home_tm,
            'mean': home_mean,
            'std': home_std
        }
        visitor_stats = {
            'transition_matrix': visitor_tm,
            'mean': visitor_mean,
            'std': visitor_std
        }
        # select a random event to start the simulation from, potentially using the use_fourth_quarter argument to make simulations shorter and speed them up
        if use_fourth_quarter:
            # Choose a 4th-quarter event with nonzero time remaining (> 30 sec)
            game_q4 = game_df[(game_df['PERIOD'] == 4) & (game_df['TIME_REMAINING'] > 30)]
            if game_q4.empty:
                continue
            event_row = game_q4.sample(n=1, random_state=random_state).iloc[0]
        else:
            event_row = game_df.sample(n=1, random_state=random_state).iloc[0]
        # Build the game_state dictionary.
        game_state = {
            'PERIOD': int(event_row['PERIOD']),
            'TIME_REMAINING': int(parse_time(event_row['PCTIMESTRING'])),
            'SCOREMARGIN_NUM': float(event_row.get('SCOREMARGIN_NUM', 0)),
            'current_event': event_row.get('SCOREUPDATEEVENT', 'NO_SHOT'),
            'score_A': None,
            'score_B': None
        }
        # run monte carlo simulation based on transition matrix for this game state
        wins = 0
        for _ in range(n_simulations):
            (score_A, score_B), _ = simulate_game_dynamic_team(game_state, home_stats, visitor_stats, avg_possession_time_fallback=avg_possession_time_fallback)
            if score_A > score_B:
                wins += 1
        model_win_pct = 100.0 * wins / n_simulations

        # pull actual outcome from play_by_play_df
        actual_outcome = get_actual_outcome_from_game(game_df, game_id)

        # determine if model matches with actual outcome 
        if (model_win_pct >= 50 and actual_outcome == 1) or (model_win_pct < 50 and actual_outcome == 0):
            prediction_status = "CORRECT"
        else: 
            prediction_status = "INCORRECT"

        # append result
        results.append({
            'GAME_ID': game_id,
            'GAME_DATE': game_date,
            'HOME_TEAM_ID': home_team_id,
            'VISITOR_TEAM_ID': visitor_team_id,
            'RANDOM_EVENTNUM': event_row['EVENTNUM'],
            'MODEL_WIN_PCT': model_win_pct,
            'ACTUAL_OUTCOME': actual_outcome,
            'PREDICTION': prediction_status,
            'HOME_MEAN_POSSESSION': home_mean,
            'HOME_STD_POSSESSION': home_std,
            'VISITOR_MEAN_POSSESSION': visitor_mean,
            'VISITOR_STD_POSSESSION': visitor_std
        })

    return pd.DataFrame(results)

# For example, evaluate 500 random games 
eval_df=evaluate_multiple_games_generic(play_by_play_df, n_games=200, n_simulations=100, avg_possession_time_fallback=18, random_state=42, use_fourth_quarter=True)
eval_df
eval_df['PREDICTION'].value_counts(normalize=True)
### 58% success rate. Slightly better than a coin flip, but might intuitively be able to do better by guessing while watching the game
## in other words, it's still pretty garbage
## it also runs really really slow

#let's test the time difference it takes to run the two models
# Model 1
start1 = time.perf_counter()
evaluate_multiple_games(play_by_play_df, transition_matrix, n_games=10, n_simulations=100, avg_possession_time=22, random_state=36)
t_model1 = time.perf_counter() - start1
t_model1

# Model 2
start2=time.perf_counter()
evaluate_multiple_games_generic(play_by_play_df, n_games=10, n_simulations=100, avg_possession_time_fallback=18, random_state=42, use_fourth_quarter=True)
t_model2 = time.perf_counter() - start2
t_model2

t_model2/t_model1



### Visualize the simulated possession times over 20 games
# Choose a team 
team_id = get_team_id('LAL')
simulation_date = '2023-03-01'  # just picking a random date

# Get possession times (in seconds) for the chosen team over multiple games
# This function groups games and computes time differences between successive shot events.
possession_times = get_team_possession_times(play_by_play_df, team_id, simulation_date)

# Filter out outlier values (e.g., >30 seconds) that might be due to clock resets or data quirks
valid_times = possession_times[possession_times < 30]

# Histogram with KDE and an overlaid normal distribution fit
plt.figure(figsize=(10, 6))
sns.histplot(valid_times, kde=True, stat="density", bins=30, color="skyblue", edgecolor="black")

# Compute mean and standard deviation of the sample
mean_val = np.mean(valid_times)
std_val = np.std(valid_times)

# Plot the fitted normal curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean_val, std_val)
plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit\nμ={mean_val:.2f}, σ={std_val:.2f}')
plt.title("Distribution of Simulated Possession Times (LAL)")
plt.xlabel("Simulated Possession Time (seconds)")
plt.ylabel("Density")
plt.legend()
plt.show()

# Generate a Q-Q plot to examine normality
plt.figure(figsize=(8, 5))
stats.probplot(valid_times, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot of Possession Times (BOS)")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()

########## Regroup - MODEL 3: let's try Semi-Markov Chains to improve the possession time estimates 


# Suppress warnings for cleaner output (optional)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="SettingWithCopyWarning") 

# Define the states based on your SCOREUPDATEEVENT
ALL_STATES = ['SHOT_2PT_MAKE', 'SHOT_3PT_MAKE', 'SHOT_FT_MAKE',
              'SHOT_2PT_MISS', 'SHOT_3PT_MISS', 'SHOT_FT_MISS', 'NO_SHOT']

# Need to be more strategic with possession times and fitting them to a distribution more thoughtful than the normal distribution
def get_team_possession_times_raw(df, team_id, simulation_date):
    #input - play_by_play_df and team_id and simulation_date
    #output - array of time between shots for a team in seconds

    # Filter for field goal attempts (made or missed)
    fg_df = df[df['EVENTMSGTYPE'].isin([1, 2])].copy()

    #filter for events for team
    team_fg_mask = (fg_df['PLAYER1_TEAM_ID'] == team_id) 
    team_fg = fg_df[team_fg_mask].copy()

    if 'GAME_DATE' not in team_fg.columns:
        print("Warning: GAME_DATE column missing for possession time calculation.")
        return np.array([]) # Return empty if no date info
    
    team_fg['GAME_DATE'] = pd.to_datetime(team_fg['GAME_DATE'], errors='coerce')
    team_fg = team_fg.dropna(subset=['GAME_DATE'])
    team_fg = team_fg[team_fg['GAME_DATE'] < simulation_date]

    if 'TIME_REMAINING' not in team_fg.columns:
            if 'PCTIMESTRING' in team_fg.columns:
                team_fg['TIME_REMAINING'] = team_fg['PCTIMESTRING'].apply(parse_time)
            else:
                print("Warning: TIME_REMAINING or PCTIMESTRING column missing.")
                return np.array([])

    team_fg = team_fg.dropna(subset=['TIME_REMAINING'])
    team_fg = team_fg.sort_values(['GAME_ID', 'PERIOD', 'TIME_REMAINING'], ascending=[True, True, False])

    possession_times = []

    # Group by game and period to calculate time diffs within the same period
    for _, group in team_fg.groupby(['GAME_ID', 'PERIOD']):
        # Calculate time difference between consecutive shots within the same period
        time_diffs = -group['TIME_REMAINING'].diff() 

        # Filter: positive difference (clock ran down), non-zero, and realistic range (e.g., <= 24 seconds)
        valid_diffs = time_diffs[(time_diffs > 0) & (time_diffs <= 24)] # Added upper bound
        possession_times.extend(valid_diffs.tolist())

    return np.array(possession_times)

#now build a function to fit the time between shots into a distribution 
#this function uses Kolmogorov-Smirnov test statistic to fit to a distribution (availabile in scipy library)
def fit_holding_time_distribution(times, dist_names=['gamma', 'weibull_min', 'lognorm']):
    #input - array of time between shots, possible distributions to use
    #output - name of the best fitting distribution, parameters for that distribution, and the KS statistic

    #return a normal distribution with mean and std if there are fewer than 10 shots for analysis 
    if len(times) < 10:
        print("Warning: Not enough data to fit a distribution.")
        return 'norm', {'loc': mean, 'scale': std}, np.inf
    #initialize the three things we are after as 0 (or infinity for the ks stat)
    best_dist_name = None
    best_params = None
    best_ks_stat = np.inf

    #loop through each of the possible distributions
    for dist_name in dist_names:
        try:
            dist = getattr(stats, dist_name)
            # Fit distribution
            if dist_name in ['gamma', 'weibull_min', 'expon']: # Distributions often defined for positive values
                 # floc=0 fixes location (minimum value) to 0. Important for durations.
                 params_tuple = dist.fit(times, floc=0)
            else: # e.g., lognorm, norm (allow location to be fitted)
                 params_tuple = dist.fit(times)

            # Perform KS test against the *fitted* distribution
            D, p_value = stats.kstest(times, dist_name, args=params_tuple)

            # Update best fit if current KS statistic is lower
            if D < best_ks_stat:
                best_ks_stat = D
                best_dist_name = dist_name
                best_params_tuple = params_tuple

        except Exception as e:
            # print(f"Warning: Could not fit {dist_name} distribution: {e}") # Optional: uncomment for debugging fits
            continue # Try the next distribution

        # --- Fallback 2: No distribution fitted successfully ---
        if best_dist_name is None:
            mean_val = np.mean(times) # Should be safe now after initial checks
            std_val = np.std(times)
            # Ensure std_val is usable (not NaN and positive)
            if np.isnan(std_val) or std_val < 1e-6:
                std_val = 3.0 # Default if std dev is invalid

            print(f"Warning: Failed to fit any specified distributions. Using Normal({mean_val:.2f}, {std_val:.2f}).")
            # Return parameters in dictionary format
            return 'norm', {'loc': mean_val, 'scale': std_val}, np.inf

    return best_dist_name, best_params, best_ks_stat

# this builds to calculating time between shots for a team
def get_weighted_team_holding_time_distribution(df, team_id, simulation_date):

    #simulation date today if for some reason it's not in the data, else convert to datetime
    if simulation_date is None:
        simulation_date = pd.Timestamp('today') + pd.Timedelta(days=1)
    else:
        simulation_date = pd.to_datetime(simulation_date)

    #use same logic as above to get relevant game ids
    team_df_mask = (df['PLAYER1_TEAM_ID'] == team_id) | (df['PLAYER2_TEAM_ID'] == team_id) | (df['PLAYER3_TEAM_ID'] == team_id)
    team_df = df[team_df_mask].copy()
    if 'GAME_DATE' not in team_df.columns: return 'norm', {'loc': 18, 'scale': 3}, np.inf # Fallback
    team_df['GAME_DATE'] = pd.to_datetime(team_df['GAME_DATE'], errors='coerce')
    team_df = team_df.dropna(subset=['GAME_DATE'])
    team_df = team_df[team_df['GAME_DATE'] < simulation_date]

    #fall back to normal distribution 
    if team_df.empty: return 'norm', {'loc': 18, 'scale': 3}, np.inf # Fallback

    #filter to last 25 games played
    team_games = team_df[['GAME_ID', 'GAME_DATE']].drop_duplicates().sort_values('GAME_DATE', ascending=True)
    recent_game_ids = list(team_games['GAME_ID'].tail(25))

    if len(recent_game_ids) == 0: return 'norm', {'loc': 18, 'scale': 3}, np.inf # Fallback

    #split into groups to weight with most recent being the most important weight
    if len(recent_game_ids) < 5:
        groups = [recent_game_ids]
        weights = [1.0]
    elif len(recent_game_ids) < 10:
        groups = [recent_game_ids[-5:], recent_game_ids[:-5]]
        weights = [0.6, 0.4]
    else:
        groups = [recent_game_ids[-5:], recent_game_ids[-10:-5], recent_game_ids[-25:-10]]
        weights = [0.5, 0.25, 0.25]

    #initialize empty lists to store the possession times and weights
    all_times = []
    data_weights = [] 

  # Collect possession times from each group and assign weights
    for game_ids, weight in zip(groups, weights):
        if not game_ids: continue
        sub_df = df[df['GAME_ID'].isin(game_ids)]
        times = get_team_possession_times_raw(sub_df, team_id, simulation_date) # Use function from above
        if len(times) > 0:
            all_times.extend(times)
            # Assign the group weight to each time point from that group, weighting data before fitting the distribution
            # Another approach is to fit separately and average parameters (more complex).
            data_weights.extend([weight] * len(times))

    if not all_times:
        print(f"Warning: No possession time data found for team {team_id} before {simulation_date}. Using defaults.")
        return 'norm', {'loc': 18, 'scale': 3}, np.inf
    
    best_dist, best_params, ks_stat = fit_holding_time_distribution(np.array(all_times))

    return best_dist, best_params, ks_stat

# Let's try it out 
nba_teams = teams.get_teams()
simulation_date = '2021-01-17'

for team in nba_teams:
    team_id = team['id']
    team_name = team['full_name']
    best_dist, best_params, ks_stat = get_weighted_team_holding_time_distribution(play_by_play_df, team_id, simulation_date)
    print(f"Best distribution for {team_name} (ID: {team_id}) before {simulation_date}: {best_dist}")
    print("Parameters:", best_params)
    print("KS Statistic:", ks_stat)

#it works! 
#tried it for all teams on multiple dates and it always comes back with lognorm distribution
#with this insight, maybe we can simplify it and always use lognorm... update - there are some different structure games (ie: All star games) that don't fit the lognorm but just the normal distribution  

#now, modify the simulation function to be driven by the new possession distribution 
def simulate_smp_game(game_state, home_stats, visitor_stats, avg_possession_time_fallback=18, min_possession_time=3):
    
    #set parameters of time remaining and possession
    period = game_state.get('PERIOD', 4) # Default to Q4 if not specified
    time_remaining_period = game_state.get('TIME_REMAINING', 0)
    # slight leap in logic here, this doesn't account for OT games, just assuming all games end after 4 quarters 
    periods_left = max(0, 4 - period)
    time_remaining = time_remaining_period + periods_left * 12 * 60
    current_state = game_state.get('current_event', 'NO_SHOT')
    team_turn = 'home'

    #initialize scores
    score_A = game_state.get('score_A')
    score_B = game_state.get('score_B')
    if score_A is None or score_B is None:
        margin = game_state.get('SCOREMARGIN_NUM', 0)
        if pd.isna(margin): margin = 0
        # Simplified score initialization based on margin (assuming margin = home - visitor)
        score_A = max(0, margin) # Simplified: Assign margin points if positive
        score_B = max(0, -margin) # Simplified: Assign margin points if negative

    #blank list to store the event sequence
    event_sequence = []

    # Helper to sample next state
    def sample_next_event(transition_matrix, current_state):
        if current_state not in transition_matrix.index:
            # If the current state is unknown, transition from a default state of "NO_SHOT" since most events are non-shooting events
            # For simplicity, transition from 'NO_SHOT' as a fallback
             current_state = 'NO_SHOT'
             if current_state not in transition_matrix.index: 
                 return "NO_SHOT" 

        # check that probabilities are valid, adding up to 1 in the transition matrix
        probs = transition_matrix.loc[current_state].values.astype(float)
        probs = np.nan_to_num(probs) # Replace NaN with 0
        total_prob = probs.sum()

        if total_prob <= 0:
             # If no valid transitions from this state, maybe end possession or default
            return "NO_SHOT"
        
        norm_probs = probs / total_prob
        possible_states = transition_matrix.columns.tolist()
        try:
            return np.random.choice(possible_states, p=norm_probs)
        except ValueError as e:
            print(f"Error sampling event from state '{current_state}'. Probs: {norm_probs}. Sum: {norm_probs.sum()}. Error: {e}")
            # Fallback if probabilities don't sum to 1 due to float issues
            # Redistribute probabilities slightly or choose randomly
            return np.random.choice(possible_states)
        
    # Helper to sample holding time
    def sample_holding_time(holding_dist_info):
        dist_name, params = holding_dist_info
        try:
            dist = getattr(stats, dist_name)
            # Sample 1 value from the distribution with the fitted parameters
            sampled_time = dist.rvs(*params[:-2], loc=params[-2], scale=params[-1], size=1)[0] if isinstance(params, tuple) \
                      else dist.rvs(loc=params.get('loc', 0), scale=params.get('scale', 1), size=1)[0] # Handle dict params from fit

            # Ensure time is positive and above minimum
            return max(min_possession_time, sampled_time)
        except Exception as e:
            print(f"Warning: Could not sample from {dist_name} with params {params}. Using fallback. Error: {e}")
            # Sample from fallback normal if specific distribution fails
            mean_fallback = params.get('loc', avg_possession_time_fallback) if isinstance(params, dict) else avg_possession_time_fallback
            std_fallback = params.get('scale', 3) if isinstance(params, dict) else 3
            fallback_time = np.random.normal(mean_fallback, std_fallback)
            return max(min_possession_time, fallback_time)

    # keep simulating and changing possession as appropriate while there's still time on the clock
    while time_remaining > 0:
        # Select stats based on whose turn it is
        if team_turn == 'home':
            tm = home_stats['transition_matrix']
            holding_dist_info = home_stats.get('holding_dist', ('norm', {'loc': 18, 'scale': 3})) # Use default if missing
        else: # Visitor's turn
            tm = visitor_stats['transition_matrix']
            holding_dist_info = visitor_stats.get('holding_dist', ('norm', {'loc': 18, 'scale': 3}))

        # Sample next event (end state of the current possession)
        next_event = sample_next_event(tm, current_state)

        #sample holding time based on the outcome of that event rather than the event that preceeded it
        consumed_time = sample_holding_time(holding_dist_info)

        # Ensure time doesn't exceed remaining time
        consumed_time = min(consumed_time, time_remaining)

        # Update time remaining in the game
        time_remaining -= consumed_time

        # Update score based on the event and whose possession it was
        points = outcome_to_points(next_event)
        if team_turn == 'home':
            score_A += points
        else:
            score_B += points

        # Record event to the list, later converted to a dataframe
        event_sequence.append({
            'team': team_turn,
            'event': next_event,
            'duration': consumed_time,
            'score_A': score_A,
            'score_B': score_B,
            'time_remaining': time_remaining
            })

        # Update current state for the *next* iteration's transition sampling
        current_state = next_event

        # Switch possession/turn 
        team_turn = 'visitor' if team_turn == 'home' else 'home'

        # Break if time runs out, effectively ending the game 
        if time_remaining <= 0:
            break

    return (score_A, score_B), event_sequence


#update evaluation function 
def evaluate_multiple_games_smp(df, n_games=100, n_simulations=100, avg_possession_time_fallback=18, random_state=42, use_fourth_quarter=False):
    # Pre-process necessary columns once
    if 'GAME_DATE' not in df.columns: df['GAME_DATE'] = pd.to_datetime(df['GAME_DATES'], errors='coerce') # Assuming GAME_DATES exists if GAME_DATE doesn't
    else: df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

    if 'SCOREUPDATEEVENT' not in df.columns:
        print("Calculating SCOREUPDATEEVENT...")
        df['SCOREUPDATEEVENT'] = df.apply(determine_shot_type_from_row, axis=1)
    if 'TIME_REMAINING' not in df.columns:
        print("Calculating TIME_REMAINING...")
        df['TIME_REMAINING'] = df['PCTIMESTRING'].apply(parse_time)
    if 'SCOREMARGIN_NUM' not in df.columns and 'SCOREMARGIN' in df.columns:
         df['SCOREMARGIN_NUM'] = pd.to_numeric(df['SCOREMARGIN'].str.replace('TIE', '0').str.replace(' ', ''), errors='coerce')

    df = df.dropna(subset=['GAME_DATE', 'TIME_REMAINING', 'SCOREUPDATEEVENT']) # Drop rows missing essential info


    np.random.seed(random_state)
    results = []

    unique_game_ids = df['GAME_ID'].unique()
    if len(unique_game_ids) == 0:
        print("No valid games found in the dataframe after processing.")
        return pd.DataFrame()

    if len(unique_game_ids) < n_games:
        n_games = len(unique_game_ids)
        print(f"Warning: Requested {n_games} games, but only {len(unique_game_ids)} available.")

    sample_game_ids = np.random.choice(unique_game_ids, size=n_games, replace=False)

    # Helper function to get team IDs 
    def determine_home_and_visitor_team_ids(game_df):
        # Try jump ball first
        jump_ball = game_df[game_df['EVENTMSGTYPE'] == 10].sort_values('EVENTNUM').iloc[0:1]
        if not jump_ball.empty:
            # Assume P1 is home, P2 is visitor at jump ball (how it's commonly done, I didn't see any instances where it was different)
            p1_team = jump_ball.iloc[0]['PLAYER1_TEAM_ID']
            p2_team = jump_ball.iloc[0]['PLAYER2_TEAM_ID']
            # Heuristic: Often the first team listed in game logs is home
            # Let's try to find the first event with a home description
            first_home_event = game_df[game_df['HOMEDESCRIPTION'].notna()].sort_values('EVENTNUM').iloc[0:1]
            if not first_home_event.empty:
                 home_id = first_home_event.iloc[0]['PLAYER1_TEAM_ID'] # Guessing P1 is the actor
                 # If the jump ball participants match this home ID, assign roles
                 if home_id == p1_team: return p1_team, p2_team
                 if home_id == p2_team: return p2_team, p1_team
            # Fallback: return jump ball teams, maybe guessing p1 is home
            print(f"Warning: Could not definitively determine home/visitor for Game {game_df['GAME_ID'].iloc[0]}. Assigning based on jump ball.")
            return p1_team, p2_team 

        # Fallback if no jump ball: Find first event with team info
        first_event = game_df[game_df['PLAYER1_TEAM_ID'].notna()].sort_values('EVENTNUM').iloc[0:1]
        if not first_event.empty:
             team1 = first_event.iloc[0]['PLAYER1_TEAM_ID']
             # Find another event with a *different* team ID
             other_event = game_df[(game_df['PLAYER1_TEAM_ID'].notna()) & (game_df['PLAYER1_TEAM_ID'] != team1)].iloc[0:1]
             if not other_event.empty:
                  team2 = other_event.iloc[0]['PLAYER1_TEAM_ID']
                  print(f"Warning: No jump ball for Game {game_df['GAME_ID'].iloc[0]}. Inferring teams {team1}, {team2}.")
                  # Cannot reliably determine home/visitor here, return in arbitrary order
                  return team1, team2 # Order might be wrong!
             else: # Only one team found?
                  print(f"Error: Only one team ID found for Game {game_df['GAME_ID'].iloc[0]}.")
                  return None, None
        else: # No team info at all?
             print(f"Error: No team IDs found for Game {game_df['GAME_ID'].iloc[0]}.")
             return None, None


    # Loop through sampled games
    for idx, game_id in enumerate(sample_game_ids, start=1):
        print(f"Processing game {idx}/{n_games} (ID: {game_id})...")
        game_df = df[df['GAME_ID'] == game_id].copy().sort_values('EVENTNUM')

        if game_df.empty:
            print(f"Skipping Game {game_id}: No data after initial filtering.")
            continue

        try:
            game_date = game_df['GAME_DATE'].iloc[0]
        except IndexError:
            print(f"Skipping Game {game_id}: Cannot determine game date.")
            continue


        # Determine Home/Visitor IDs
        home_team_id, visitor_team_id = determine_home_and_visitor_team_ids(game_df)
        if home_team_id is None or visitor_team_id is None:
             print(f"Skipping Game {game_id}: Could not determine teams.")
             continue

        print(f"  Game Date: {game_date.date()}, Home ID: {home_team_id}, Visitor ID: {visitor_team_id}")

        # Build SMP models for each team
        print("  Building Home Team Model...")
        home_tm = build_weighted_team_transition_matrix(df, home_team_id, state_col='SCOREUPDATEEVENT', simulation_date=game_date)
        home_dist, home_params, _ = get_weighted_team_holding_time_distribution(df, home_team_id, simulation_date=game_date)
        home_stats = {
            'transition_matrix': home_tm,
            'holding_dist': (home_dist, home_params)
        }
        print(f"  Home Holding Dist: {home_dist} {home_params}")


        print("  Building Visitor Team Model...")
        visitor_tm = build_weighted_team_transition_matrix(df, visitor_team_id, state_col='SCOREUPDATEEVENT', simulation_date=game_date)
        visitor_dist, visitor_params, _ = get_weighted_team_holding_time_distribution(df, visitor_team_id, simulation_date=game_date)
        visitor_stats = {
            'transition_matrix': visitor_tm,
            'holding_dist': (visitor_dist, visitor_params)
        }
        print(f"  Visitor Holding Dist: {visitor_dist} {visitor_params}")


        # Select random event to start simulation from
        if use_fourth_quarter:
            eligible_events = game_df[(game_df['PERIOD'] >= 4) & (game_df['TIME_REMAINING'] > 30) & game_df['SCOREMARGIN_NUM'].notna()]
            if eligible_events.empty:
                 # Fallback to any event if no suitable 4th quarter event found
                 eligible_events = game_df[game_df['TIME_REMAINING'] > 0 & game_df['SCOREMARGIN_NUM'].notna()]
                 if eligible_events.empty:
                     print(f"Skipping Game {game_id}: No suitable start event found.")
                     continue
            start_event_row = eligible_events.sample(n=1, random_state=random_state).iloc[0]
        else:
            eligible_events = game_df[game_df['TIME_REMAINING'] > 0 & game_df['SCOREMARGIN_NUM'].notna()]
            if eligible_events.empty:
                 print(f"Skipping Game {game_id}: No suitable start event found.")
                 continue
            start_event_row = eligible_events.sample(n=1, random_state=random_state).iloc[0]

        # Prepare initial game state for simulation
        initial_game_state = {
            'PERIOD': int(start_event_row['PERIOD']),
            'TIME_REMAINING': int(start_event_row['TIME_REMAINING']),
            'SCOREMARGIN_NUM': float(start_event_row.get('SCOREMARGIN_NUM', 0)), # Use get with default
            'current_event': start_event_row.get('SCOREUPDATEEVENT', 'NO_SHOT'), # Use get with default
            'score_A': None, # Let simulation calculate from margin initially
            'score_B': None
        }
        print(f"  Starting simulation from Event {start_event_row['EVENTNUM']} (Period {initial_game_state['PERIOD']}, Time {initial_game_state['TIME_REMAINING']}s, Margin {initial_game_state['SCOREMARGIN_NUM']}, Last Event '{initial_game_state['current_event']}')")

        # Run simulations
        wins_A = 0
        print(f"  Running {n_simulations} simulations...")
        for i in range(n_simulations):
            (final_score_A, final_score_B), _ = simulate_smp_game(
                initial_game_state,
                home_stats,
                visitor_stats,
                avg_possession_time_fallback=avg_possession_time_fallback
            )
            if final_score_A > final_score_B:
                wins_A += 1
        model_win_pct_A = 100.0 * wins_A / n_simulations
        print(f"  Simulation Result: Home Win Pct = {model_win_pct_A:.2f}%")

        # Get actual outcome
        actual_outcome_A = get_actual_outcome_from_game(game_df, game_id) # 1 if A wins, 0 if B wins (or ties)
        print(f"  Actual Outcome: {'Home Win' if actual_outcome_A == 1 else 'Visitor Win/Tie'} ({actual_outcome_A})")


        # Determine prediction status
        prediction_correct = np.nan
        if actual_outcome_A is not None:
             predicted_outcome_A = 1 if model_win_pct_A >= 50 else 0
             prediction_correct = 1 if predicted_outcome_A == actual_outcome_A else 0
             print(f"  Prediction: {'CORRECT' if prediction_correct == 1 else 'INCORRECT'}")

        # Append results
        results.append({
            'GAME_ID': game_id,
            'GAME_DATE': game_date.date(),
            'HOME_TEAM_ID': home_team_id,
            'VISITOR_TEAM_ID': visitor_team_id,
            'STARTING_EVENTNUM': start_event_row['EVENTNUM'],
            'STARTING_PERIOD': initial_game_state['PERIOD'],
            'STARTING_TIME_REM': initial_game_state['TIME_REMAINING'],
            'STARTING_MARGIN': initial_game_state['SCOREMARGIN_NUM'],
            'MODEL_HOME_WIN_PCT': model_win_pct_A,
            'ACTUAL_HOME_WIN': actual_outcome_A,
            'PREDICTION_CORRECT': prediction_correct,
            'HOME_HOLDING_DIST': home_dist,
            'HOME_HOLDING_PARAMS': home_params,
            'VISITOR_HOLDING_DIST': visitor_dist,
            'VISITOR_HOLDING_PARAMS': visitor_params
        })

    return pd.DataFrame(results)


smp_results = evaluate_multiple_games_smp(
    df=play_by_play_df,
    n_games=400,          # Number of games to test
    n_simulations=100,   # Simulations per game start point
    use_fourth_quarter=True 
)
smp_results['PREDICTION_CORRECT'].value_counts(normalize=True)
smp_results['STARTING_PERIOD'].value_counts()
smp_results.to_csv('/Users/collinsch/Downloads/results.csv')
#plot out the minutes and score margin to test the rule based on the simulation 
results_df = smp_results.copy()
results_df['minutes_left'] = (results_df['STARTING_TIME_REM'] // 60).astype(int)
agg = (
    results_df
    .groupby(['STARTING_MARGIN','minutes_left'])
    ['MODEL_HOME_WIN_PCT']
    .apply(lambda pct: (pct == 100).mean() * 100)
    .reset_index(name='pct_model_100')
)
heat = agg.pivot(
    index='STARTING_MARGIN',
    columns='minutes_left',
    values='pct_model_100'
)
labels = heat.copy().astype(object)
for margin in heat.index:
    for mins in heat.columns:
        val = heat.at[margin, mins]
        if pd.isna(val):
            labels.at[margin, mins] = "N/A"
        elif val == 100:
            labels.at[margin, mins] = "W"
        elif val == 0:
            labels.at[margin, mins] = "L"
        else:
            labels.at[margin, mins] = ""
plt.figure(figsize=(10, 6))
mask = heat.isna()
sns.heatmap(
    heat, mask=mask, annot=labels, fmt="", cmap="YlGnBu",
    cbar=False, linewidths=0.5, linecolor="gray",
    vmin=0, vmax=100   # keeps your color scale consistent
)
plt.title("Model’s 100% Certainty (W=100%, L=0%, N/A=not simulated)")
plt.xlabel("Minutes Remaining")
plt.ylabel("Score Margin (Home − Visitor)")
plt.tight_layout()
plt.show()





### Simulate just Lakers games
lakers_id   = 1610612747   
cutoff_date = '2025-03-14'
holding_dist_name, holding_params, _ = get_weighted_team_holding_time_distribution(
    play_by_play_df,
    lakers_id,
    cutoff_date
)

# 3) Now wrap these into the dict structure expected by simulate_smp_game:
P_lakers, F_lakers = build_team_semi_markov(
    play_by_play_df,
    team_id    = lakers_id,
    date_cutoff= cutoff_date,
    state_col  = 'SCOREUPDATEEVENT'   # ← this gives you states like 'SHOT_2PT_MAKE'
)

lakers_stats = {
    'transition_matrix': P_lakers,
    'holding_dist':      (holding_dist_name, holding_params)
}

#function from above
def outcome_to_points(evt):
    return {
        'SHOT_2PT_MAKE': 2,
        'SHOT_3PT_MAKE': 3,
        'SHOT_FT_MAKE':  1
    }.get(evt, 0)

#establish simulations to run
minutes_left = range(1,13)     # 1 to 12 minutes remaining in 4th
margins      = range(0,12)     # Lakers up by 0 to 12 points
n_sims       = 500

rows = []
valid_states = list(P_lakers.index)    # these are ints, not strings
for mins in minutes_left:
  for m in margins:
    wins = 0
    for _ in range(n_sims):
      gs = {
        'PERIOD': 4,
        'TIME_REMAINING': mins * 60,
        'current_event': np.random.choice(valid_states),  # ← pick a real state
        'score_A': m,
        'score_B': 0
      }
      (sA, sB), _ = simulate_smp_game(gs, lakers_stats, lakers_stats)
      if sA > sB: wins += 1

    rows.append({
            'minutes_left': mins,
            'margin':       m,
            'win_pct':      wins / n_sims * 100
        })

grid_df = pd.DataFrame(rows)

grid_df

grid_df.to_csv('/Users/collinsch/Downloads/lakers_sim.csv')

grid_df['predicted_win'] = grid_df['win_pct'] >= 50

# Pivot to a margin × minutes grid using win_pct
pivot_win = grid_df.pivot(index='margin', columns='minutes_left', values='win_pct')

# Build annotations by formatting win_pct values as percentages
labels = pivot_win.copy().astype(object)
for margin in pivot_win.index:
    for mins in pivot_win.columns:
        val = pivot_win.at[margin, mins]
        if pd.isna(val):
            labels.at[margin, mins] = ""
        else:
            labels.at[margin, mins] = f"{val:.1f}%"

# Plot the heatmap with smaller annotations
plt.figure(figsize=(8, 6))
mask = pivot_win.isna()
sns.heatmap(
    pivot_win.astype(float),
    mask=mask,
    annot=labels,
    fmt="",
    cmap="viridis",
    cbar=True,
    linewidths=0.5,
    linecolor="gray",
    square=True,
    annot_kws={"size": 9, "fontfamily": "monospace"}
)
plt.title("Lakers Win Percentage by Margin & Minutes Left")
plt.xlabel("Minutes Remaining in Q4")
plt.ylabel("Score Margin (Lakers − Opponent)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


condition=grid_df[grid_df['margin']>grid_df['minutes_left']]
condition['predicted_win'].value_counts(normalize=True)
condition['win_pct'].mean()

#only one point higher average winning percentage
subset = grid_df[grid_df['margin'] == grid_df['minutes_left'] + 1]
avg_win_pct = subset['win_pct'].mean()
print("Average win percentage (margin is one point higher than minutes left):", avg_win_pct)


















#### Try a new model 3 ####################################################
# This model totally failed. Only had 53% accuraccy. So much time for worse performance. 
def build_team_semi_markov(play_by_play_df, team_id, date_cutoff, state_col='EVENTMSGTYPE'):
    df = play_by_play_df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df[df['GAME_DATE'] < pd.to_datetime(date_cutoff)]
    # Filter to events where the team is involved
    df = df[(df['PLAYER1_TEAM_ID'] == team_id) | (df['PLAYER2_TEAM_ID'] == team_id)]

    # Keep most recent 25 games
    recent_games = (
        df[['GAME_ID', 'GAME_DATE']]
          .drop_duplicates()
          .sort_values('GAME_DATE')
          .tail(25)['GAME_ID']
    )
    df = df[df['GAME_ID'].isin(recent_games)].sort_values(['GAME_ID','EVENTNUM'])

    # Count transitions and collect dwelling times
    transitions = {}
    hold_times = {}
    for _, grp in df.groupby('GAME_ID'):
        states = grp[state_col].values
        times  = grp['TIME_REMAINING'].astype(float).values
        for i in range(len(states)-1):
            s0, s1 = states[i], states[i+1]
            transitions[(s0,s1)] = transitions.get((s0,s1), 0) + 1
            dt = times[i] - times[i+1]
            if dt > 0:
                hold_times.setdefault((s0,s1), []).append(dt)

    # Build transition probability matrix P
    all_states = sorted({s for (s,_) in transitions} | {t for (_,t) in transitions})
    P = pd.DataFrame(0.0, index=all_states, columns=all_states)
    row_totals = {}
    for (s0,s1), cnt in transitions.items():
        row_totals[s0] = row_totals.get(s0,0) + cnt
    for (s0,s1), cnt in transitions.items():
        P.at[s0,s1] = cnt / row_totals[s0]

    # Fit lognormal distribution to holding times for each transition
    F = {}
    for key, times in hold_times.items():
        # force loc=0 for lognormal
        shape, loc, scale = stats.lognorm.fit(times, floc=0)
        F[key] = (shape, loc, scale)
    return P, F

#test out the function 
build_team_semi_markov(play_by_play_df, team_id, date_cutoff='2023-03-01', state_col='SCOREUPDATEEVENT')
#works, returns dictionary with transition state, the sigma/shape, loc is forced to 0, and the median/scale for the lognormal distribution

#build a function to simulate next game event 

MISS_CODES  = {2}    # missed field goal
TURNOVERS   = {5}    # turnover
REBOUNDS    = {4}    # rebound

SCORE_MAP = {1:2, 3:3, 6:1}  # 1=2pt make,3=3pt make,6=FT

def outcome_to_points(evt: int) -> int:
    return SCORE_MAP.get(evt, 0)

def simulate_smp_game(game_state, home_model, vis_model):
    #holding time dictionary for the home team
    P_home, F_home = home_model
    #holding time dictionary for the visitor team
    P_vis,  F_vis  = vis_model

    # compute total seconds remaining for the rest of the game
    period = game_state['PERIOD'] #for this project, it's always going to be fourth quarter, but we'll calculate this in case we want to expand in the future 
    time_remaining = game_state['TIME_REMAINING'] + max(0,4-period) * 12*60
    state = game_state['current_state'] #same dictionary as in pervious models 
    scoreA, scoreB = game_state['score_A'], game_state['score_B'] #grab score from both teams
    team = 'home'
    turn='home'
    seq = []
    # track last shooter for rebound logic
    last_shooter = None

    while time_remaining > 0:
        P,F = (P_home, F_home) if team=='home' else (P_vis, F_vis)
        # ensure valid state
        if state not in P.index:
            state = np.random.choice(P.index) #choose a random state to start in if current state isn't in index
        row = P.loc[state].values
        if row.sum()>0:
            p = row/row.sum()
            next_state = np.random.choice(P.columns, p=p)
        else:
            next_state = 0
        # dwelling time from lognormal or fallback
        key = (state, next_state)
        if key in F:
            shp, loc, sc = F[key] #splitting variables created in previous function
            dt = stats.lognorm.rvs(shp, loc=loc, scale=sc)
        else:
            dt = np.random.normal(18,3) #fallback to what we were doing in model 2
        #make dwelling time to mininmum of 1 second, and it can't exceed time remaining
        dt = np.clip(dt, 1, time_remaining)
        time_remaining -= dt
        # score update
        pts = outcome_to_points(next_state)
        if team=='home': 
            scoreA += pts
        else:           
            scoreB += pts
        seq.append((team, state, next_state, dt, scoreA, scoreB, time_remaining)) #append the next state and score change to the blank list above
        # possession change on misses/turnovers/fouls
        # possession change
        if next_state in MISS_CODES|TURNOVERS:
            turn = 'visitor' if turn=='home' else 'home'
        elif next_state in REBOUNDS:
            if last_shooter!=turn:
                turn = 'visitor' if turn=='home' else 'home'
        state = next_state
    return (scoreA, scoreB), seq


#make embedded function used in previous models more clear here
def determine_home_and_visitor_team_ids(game_df):
        # Try jump ball first
        jump_ball = game_df[game_df['EVENTMSGTYPE'] == 10].sort_values('EVENTNUM').iloc[0:1]
        if not jump_ball.empty:
            # Assume P1 is home, P2 is visitor at jump ball (common but not guaranteed)
            # Need a more robust way if this isn't reliable (e.g., check team city/abbr)
            p1_team = jump_ball.iloc[0]['PLAYER1_TEAM_ID']
            p2_team = jump_ball.iloc[0]['PLAYER2_TEAM_ID']
            # Heuristic: Often the first team listed in game logs is home
            # Let's try to find the first event with a home description
            first_home_event = game_df[game_df['HOMEDESCRIPTION'].notna()].sort_values('EVENTNUM').iloc[0:1]
            if not first_home_event.empty:
                 home_id = first_home_event.iloc[0]['PLAYER1_TEAM_ID'] # Guessing P1 is the actor
                 # If the jump ball participants match this home ID, assign roles
                 if home_id == p1_team: return p1_team, p2_team
                 if home_id == p2_team: return p2_team, p1_team
            # Fallback: return jump ball teams, maybe guessing p1 is home
            print(f"Warning: Could not definitively determine home/visitor for Game {game_df['GAME_ID'].iloc[0]}. Assigning based on jump ball.")
            return p1_team, p2_team # Default guess


def evaluate_multiple_games_smp(df: pd.DataFrame,
                                n_games: int=100,
                                n_sims:  int=100,
                                random_state: int=34):  
    np.random.seed(random_state)
    out=[]
    games = df['GAME_ID'].unique() #get all game ids
    sample = np.random.choice(games, min(n_games,len(games)), replace=False) #select a random sample of game ids, length specified by n_games
    for idx,gid in enumerate(sample):
        print(f"[{idx}/{len(sample)}] Processing GAME_ID={gid}")
        g = df[df['GAME_ID']==gid].sort_values('EVENTNUM') #sort individual game by event number
        # pick a Q4 moment with >30s left
        q4 = g[(g['PERIOD']>=4)&(g['TIME_REMAINING']>60)] #pick a random event in the fourth quarter with more than 60 seconds remaining
        if q4.empty: continue
        row = q4.sample(1).iloc[0]
        # build the dictionary for game state that will be passed into simulation function
        gs = {
            'PERIOD': int(row['PERIOD']),
            'TIME_REMAINING': float(row['TIME_REMAINING']),
            'current_state': int(row['EVENTMSGTYPE']),
            'score_A': max(0, row.get('SCOREMARGIN_NUM', 0)),
            'score_B': max(0, -row.get('SCOREMARGIN_NUM', 0))
        }
        # Determine home and visitor team IDs from the game dataframe
        #I think there's some funky all star/exhibition games that are tripping up the function
        #going to liberally skip any games where game ID can't be found
        ids = determine_home_and_visitor_team_ids(g)
        if ids is None:
            print(f"  Skipping GAME_ID {gid}: cannot determine home/visitor")
            continue
        home_team_id, visitor_team_id = ids
        # build markov models - both possession and dwelling time using the inferred team IDs
        home_mod = build_team_semi_markov(df, home_team_id, row['GAME_DATE'])
        vis_mod  = build_team_semi_markov(df, visitor_team_id, row['GAME_DATE'])
        P_home, F_home = home_mod
        P_vis,  F_vis  = vis_mod
        if P_home.empty or P_vis.empty:
            print(f"  Skipping GAME_ID {gid}: no transition data for home or visitor")
            continue
        # simulate
        wins=0
        for _ in range(n_sims):
            (sA,sB),_ = simulate_smp_game(gs, home_mod, vis_mod)
            if sA>sB: wins+=1
        pct = wins/n_sims
        actual = get_actual_outcome_from_game(g,gid)
        corr = int((pct>=0.5 and actual==1) or (pct<0.5 and actual==0))
        out.append({'GAME_ID':gid,'PCT_HOME_WIN':pct,'ACTUAL':actual,'CORRECT':corr})
    return pd.DataFrame(out)

## let's test it an evaluate the model 3 functions
smp_results = evaluate_multiple_games_smp(
    df=play_by_play_df,
    n_games=100,          # Number of games to test
    n_sims=100,   # Simulations per game start point
)

smp_results['CORRECT'].value_counts(normalize=True)