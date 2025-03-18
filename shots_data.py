"""Generates a csv of all shots from the top five leagues in the dataset."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset ------------------------------------------------------------
dir1 = ""

# read in teams.csv
filename = dir1 + r"base_data/teams.csv"
df_teams = pd.read_csv(filename)

# read in leagues.csv
filename = dir1 + r"base_data/leagues.csv"
df_leagues = pd.read_csv(filename)

# read in teamStats.csv
filename = dir1 + r"base_data/teamStats.csv"
df_teamStats = pd.read_csv(filename)

# read in fixtures.csv
filename = dir1 + r"base_data/fixtures.csv"
df_fixtures = pd.read_csv(filename)

# read in players.csv
filename = dir1 + r"base_data/players.csv"
df_players = pd.read_csv(filename)

# read in keyEventDescription.csv
filename = dir1 + r"base_data/keyEventDescription.csv"
df_keyEventDescription = pd.read_csv(filename)

# filter for top five leagues ---------------------------------------------
valid_leagues = ["English Premier League", "French Ligue 1", "Spanish LALIGA", "Italian Serie A", "German Bundesliga"]
valid_leagueIds = list(df_leagues[df_leagues['leagueName'].apply(lambda x: x in valid_leagues)]['leagueId'])
valid_leagueMidsizeNames = list(df_leagues[df_leagues['leagueName'].apply(lambda x: x in valid_leagues)]['midsizeName'])
valid_events = list(df_fixtures[df_fixtures['leagueId'].apply(lambda x: x in valid_leagueIds)]['eventId'])

# get IDs of goals and shots ------------------------------------------------
goalIds = [70, 137, 138, 173]
shotIds = [74, 106, 117, 140, 136, 135]
penalty_missIds = [174, 140, 115, 114, 113]
penalty_scoreIds = [104, 98]

# load all plays ------------------------------------------------
fname = "plays_data/plays_" + str(2024) + "_" + valid_leagueMidsizeNames[0] + ".csv"
df_plays = pd.read_csv(fname)
for mid_name in valid_leagueMidsizeNames[1:]:
    fname = "plays_data/plays_" + str(2024) + "_" + mid_name + ".csv"
    df_temp = pd.read_csv(fname)
    df_plays = pd.concat([df_plays, df_temp], ignore_index=True)
df_plays = df_plays[df_plays['eventId'].apply(lambda x: x in valid_events)]
del df_temp

# filter for shots and goals
is_goal = df_plays['typeId'].apply(lambda x: x in goalIds)
is_shot = df_plays['typeId'].apply(lambda x: x in shotIds)
df_shots = df_plays[is_shot | is_goal].copy()
df_shots['isGoal'] = is_goal

# delete duplicate playIds
df_shots = df_shots.drop_duplicates(subset=['playId'], keep='first')

# define feature extraction functions ------------------------------------
goal_y_bounds = [0.43, 0.57]
def distance_to_goal(x, y):
    """Returns distance to nearest point on the goal"""
    # rescale
    x *= 60
    y *= 80
    if y < min(goal_y_bounds):
        return np.sqrt(x**2 + (0.43*80 - y)**2) # distance to front post
    elif y > max(goal_y_bounds):
        return np.sqrt(x**2 + (y - 0.57*80)**2)
    else:
        return x

def angle_of_shot(x, y, x2, y2):
    """Returns the direction of the shot relative to pointing straight at the goalline.
    I may have flipped the x-axis >:-)
    """
    vec = np.array([x - x2, y - y2])
    goal_vec = np.array([1, 0])
    return np.arccos(vec@goal_vec/(np.linalg.norm(vec)*np.linalg.norm(goal_vec)))

def angle_to_goal(x, y):
    # rescale
    x *= 60
    y *= 80
    """Returns the angle to the nearest point on the goal"""
    if y < min(goal_y_bounds):
        return angle_of_shot(x, y, 0, 0.43*60)
    elif y > max(goal_y_bounds):
        return angle_of_shot(x, y, 0, 0.57*80)
    else:
        return 0
    
def get_athlete_position(athleteId):
    """Returns the position of the athlete"""
    return df_players[df_players['athleteId'] == athleteId]['positionAbbreviation'].values[0]

# add features
df_shots['shotDistance'] = df_shots.apply(lambda row: distance_to_goal(row['fieldpositionX'], row['fieldPositionY']), axis=1)
df_shots['shotAngle'] = df_shots.apply(lambda row: angle_to_goal(row['fieldpositionX'], row['fieldPositionY']), axis=1)
df_shots['shooterPosition'] = df_shots['athleteId'].apply(get_athlete_position)

# save to csv -----------------------------------------------
df_shots.to_csv('shots_data.csv', index=False)