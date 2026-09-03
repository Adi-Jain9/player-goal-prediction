# Player Goal Prediction

A Python scipt that uses machine-learning to predicts player goals using recent
match performance.

## Approach

- Sorts each player's matches chronologically
- Creates rolling averages from the previous five matches
- Fits a linear-regression model using NumPy least squares
- Evaluates predictions using MAE and RMSE

## Features

- Goals
- Shots
- Shots on target
- Expected goals (xG)
- Minutes played
- Player rating
- Age

## Technologies

Python and NumPy

## Project Files

- `data_preprocessing.py` — cleans the data and generates rolling features
- `linear_regression.py` — trains and evaluates the regression model

## Dataset

The dataset was obtained from Kaggle and is not included in this repository.
Download `fifa_world_cup_2026_player_performance.csv` and place it in the
project directory before running the scripts.

## Running the Project

```bash
python3 data_preprocessing.py
python3 linear_regression.py
