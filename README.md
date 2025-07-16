Overview
This repository contains code and documentation to accompany the paper “When is an NBA
Game Really Over.” The intent behind this documentation is to show how the three simulation
models were built and highlight the differences between them. This project implements three
models of increasing complexity:
- Model 1: Standard Possession Time with a Markov Chain to transition between events
- Model 2: Dynamically Calculated Possession Time with a Markov Chain to transition
between events
- Model 3: Semi Markov Chain with lognormal possession distributions
Repository Structure
├── raw_code/ # Unmodified raw scripts
├── simulation_models/ # Documented functions for each model
│ ├── model1.py
│ ├── model2.py
│ └── model3.py
│ ├── helper_functions.py
├── sample_play_by_play_df.csv # extract from the full dataset
├── Simulation Models Pseudo Code #Psuedo code/ order of operations in functions to
drive simulation
└── README.md # This documentation
Requirements
Python 3.8+
pandas
numpy
nba_api
scipy
matplotlib (for generating plots)
Data Source
- Play-by-play data via the NBA API PlayByPlayV2 endpoint (swar, n.d.)
- Coverage from 2020/2021 season through when the data was pulled 3/14/2025
Note: the data folder contains a csv of a few games of data, but not the entire dataset. The
entire dataset used was over 16 million rows of event data
Simulation Model Descriptions
Model 1 - Standard Possession Time Markov Chain
- Single transition matrix built on the entire dataset
- Fixed possession time of 20 seconds for all non free throw events
- Key functions: build_transition_matrix(), simulate_markov_chain()
Model 2 - Dynamically Calculated Possession Time and Team Specific Markov Chain
- Team specific shooting percentages weighted over the last 5, 10, and 25 games to
account for team hot streaks
- Each team fits a normal distribution for possession time per team
- Key functions: calculate_weighted_stats(), simulate_dynamic_chain()
Model 3 - Semi-Markov Chains with lognormal distributions
- Uses Kolmogorov–Smirnov tests to fit lognormal distributions to each state transition
time. Each state to state transition is associated with a lognormal distribution of times
- Simulates event sequence with state matrix P and holding time dictionary F
- Key functions: fit_possession_distributions(), simulate_semi_markov()
Plots
plots/ contains:
- lakers_predictions.png
- fg_attempts_per_min.png
- shots_per_min.png
- events_per_game.png
