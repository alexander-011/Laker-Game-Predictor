import requests
from bs4 import BeautifulSoup
import tensorflow as tf
import numpy as np

def get_team_data(team_name, teams_data, win_percentage_data, ppg_data):
    for i in range(len(teams_data)):
        if teams_data[i].text.strip() == team_name:
            win_percentage = float(win_percentage_data[i].text.strip())
            ppg = float(ppg_data[i].text.strip())
            return {'Win Percentage': win_percentage, 'PPG': ppg}
    return None

def get_nba_matchup():
    url = "https://www.statmuse.com/nba/ask/nba-teams-winning-percentage-the-last-5-years"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        teams_data = soup.find_all('td', class_='TeamName-label')
        win_percentage_data = soup.find_all('td', class_='numericValue-winPercentage')
        ppg_data = soup.find_all('td', class_='numericValue-ppg')

        lakers_data = get_team_data("Los Angeles Lakers", teams_data, win_percentage_data, ppg_data)
        
        opponent_data = get_team_data("Lakers' Opponent", teams_data, win_percentage_data, ppg_data)

        if lakers_data and opponent_data:
            lakers_win_percentage, lakers_ppg = lakers_data['Win Percentage'], lakers_data['PPG']
            opponent_win_percentage, opponent_ppg = opponent_data['Win Percentage'], opponent_data['PPG']

            team_a_stats = np.array([lakers_win_percentage, lakers_ppg])
            team_b_stats = np.array([opponent_win_percentage, opponent_ppg])

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            predicted_prob = model.predict(np.array([team_a_stats - team_b_stats]))

            
            predicted_winner = "Los Angeles Lakers" if predicted_prob > 0.5 else "Opponent"

            
            print("Predicted winner:", predicted_winner)

        else:
            print("Data not found for one or both teams.")
    else:
        print("Error", response.status_code)


get_nba_matchup()
