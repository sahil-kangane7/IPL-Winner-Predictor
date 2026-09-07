import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

MODEL_PATH = Path(__file__).resolve().parent / "pipeipl.pkl"

pipe = pickle.load(open(MODEL_PATH, 'rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target', min_value=0, step=1, format="%d")

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1, format="%d")
with col4:
    overs_completed = st.number_input('Overs completed', min_value=0, max_value=20, format="%d")
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1, format="%d")

if st.button('Predict Probability'):
    overs_int = int(overs_completed)
    balls_part = int((overs_completed - overs_int) * 10)

    runs_left = target - score
    balls_left = 120 - (overs_completed*6)
    wickets = 10 - wickets

    total_runs_x = target

    crr = score/overs_completed
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],
                             'bowling_team':[bowling_team],
                             'city':[city],
                             'runs_left':[runs_left],
                             'balls_left':[balls_left],
                             'wickets':[wickets],
                             'total_runs_x':[target],
                             'crr':[crr],
                             'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")