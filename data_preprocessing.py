import pandas as pd
import numpy as np

FILE_NAME = "fifa_world_cup_2026_player_performance.csv"

WINDOW_SIZE = 5

STATS = [
    "goals",
    "shots",
    "shots_on_target",
    "expected_goals_xg",
    "minutes_played",
    "player_rating"
]

def load_data():

    # read csv
    df = pd.read_csv(FILE_NAME)

    # convert match dates into real dates
    df["match_date"] = pd.to_datetime(df["match_date"])

    # sort each player matches chronologically
    df = df.sort_values(["player_id", "match_date", "match_id"]).reset_index(drop=True)

    
    # CREATING LAST 5 MATCH
    feature_names = []

    for stat in STATS:
        feature_name = ("avg_" + stat + "_last5")
        
        df[feature_name] = (df.groupby("player_id")[stat].transform(lambda x: x.shift(1).rolling(
            window = WINDOW_SIZE, min_periods=1).mean()))

        feature_names.append(feature_name)


    # AGE
    feature_names.append("age")

    # MARKET VALUE
    df["log_market_value"] = np.log1p(df["market_value_eur"])

    feature_names.append("log_market_value")

    # POSITION
    df["is_forward"] = (df["position"] == "Forward").astype(int)
    df["is_midfielder"] = (df["position"] == "Midfielder").astype(int)
    df["is_defender"] = (df["position"] == "Defender").astype(int)

    feature_names.extend(["is_forward", "is_midfielder", "is_defender"])


    # REMOVE ROWS THAT ARE NAN
    df = df.dropna(subset=feature_names).reset_index(drop=True)

    # GET MATCH DATES
    dates = np.sort(df["match_date"].unique())

    # DISTRIBUTING GAMES BETWEEN TRAINING AND TESTING
    train_index = int(len(dates) * 0.60)
    validation_index = int(len(dates) * 0.80)

    train_end_date = dates[train_index]

    validation_end_date = dates[validation_index]

    # Training Data
    train = df[df["match_date"] < train_end_date]

    # Validation data
    validation = df[(df["match_date"] >= train_end_date) & 
                    (df["match_date"] < validation_end_date)]

    # Test Data
    test = df[df["match_date"] >= validation_end_date]

    # FEATURES AND TARGETS
    X_train = train[feature_names].to_numpy(dtype=float)

    y_train = train["goals"].to_numpy(dtype=float)

    X_validation = validation[feature_names].to_numpy(dtype=float)

    y_validation = validation["goals"].to_numpy(dtype=float)

    X_test = test[feature_names].to_numpy(dtype=float)

    y_test = test["goals"].to_numpy(dtype = float)

    # STANDARDIZATION
    means = np.mean(X_train, axis = 0)

    standard_deviations = np.std(X_train, axis = 0)

    standard_deviations[standard_deviations == 0] = 1

    X_train_scaled = (X_train - means) / standard_deviations

    X_validation_scaled = (X_validation - means) / standard_deviations

    X_test_scaled = (X_test - means) / standard_deviations

    return ( 
        X_train,
        X_validation,
        X_test,
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        y_train,
        y_validation,
        y_test,
        feature_names,
        means,
        standard_deviations
        )
