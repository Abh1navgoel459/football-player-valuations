from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Allow all origins during development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PlayerData(BaseModel):
    PassesLeadingToGoals: int
    Age: int
    NumberOfTimesPassTarget: int
    NumberOfTimesReceivedPass: int
    TouchesInAttackingPenaltyBox: int
    TotalCarriesInForwardDirection: int
    ContractYearsLeft: int
    SuccessfulPressurePercent: int
    CarriesIntoAttackingPenaltyBox: int
    GoalCreatingActionsPer90: int
    GlsAstScoredPenaltiesPer90: int
    GoalCreatingActions: int
    GA90: int
    xG: int
    League: str
    Position: str
    Value: int  # Add this field to compare actual value

# Global variable to hold the trained model
best_gb_model = None

@app.get("/over-under-valued")
async def get_over_under_valued():
    # Load your CSV file
    csv_file = '/Users/abhinavgoel/Downloads/final_dataset.csv'
    df = pd.read_csv(csv_file)

    # Drop duplicates based on 'Player'
    df = df.drop_duplicates(subset=['Player'], keep='first')

    # Define position weights based on value hierarchy
    position_map = {
        'attack': 4,
        'midfield': 3,
        'Defender': 2,
        'Goalkeeper': 1
    }
    df['Position_Value'] = df['Position'].map(position_map)

    # Map "League" column
    league_map = {
        'Premier League': 5,
        'Bundesliga': 4,
        'Serie A': 2,
        'La Liga': 3,
        'Ligue 1': 1
    }
    df['League'] = df['League'].map(league_map)

    # Drop unwanted columns
    unwanted_columns = ['Club', 'Nation']  # Keep 'Player' for now
    df.drop(columns=unwanted_columns, inplace=True, errors='ignore')

    # Define a function to apply age and contract year weights
    def apply_age_contract_weights(df):
        # Weightage for contract years left
        df['Contract_Weight'] = df['Contract Years Left'].apply(lambda years_left: 3 if years_left >= 5 else (2 if years_left >= 2 else 1))
        return df

    # Apply age and contract weights
    df = apply_age_contract_weights(df)

    # Define weight mapping for different seasons
    season_weights = {
        '(20/21)': 1.0,
        '(19/20)': 0.93,
        '(18/19)': 0.75,
        '(17/18)': 0.58
    }

    # Function to apply season weights
    def apply_season_weights(df, season_weights):
        for season, weight in season_weights.items():
            season_cols = [col for col in df.columns if season in col]
            for col in season_cols:
                df[col] *= weight
        return df

    # Apply season weights
    df = apply_season_weights(df, season_weights)

    # List of most important features
    important_features = [
        'Passes Leading to Goals (20/21)',
        'Age',
        'Number of Times Player was Pass Target (20/21)',
        'Number of Times Received Pass (19/20)',
        'Touches in Attacking Penalty Box (20/21)',
        'Total Carries in Forward Direction (20/21)',
        'Contract Years Left',
        'Successful Pressure % (17/18)',
        'Carries into Attacking Penalty Box (20/21)',
        'Goal Creating Actions/90 (19/20)',
        '(Gls+Ast-Scored Penalties)/90 (19/20)',
        'Goal Creating Actions (20/21)',
        '(G+A)/90 (19/20)',
        '(G+A)/90 (20/21)',
        'xG (20/21)',
        'Number of Times Player was Pass Target (18/19)',
        'Number of Times Received Pass (20/21)',
        'League',
        'Number of Times Player was Pass Target (19/20)',
        'Position_Value',
        'Player'  # Add 'Player' column back to retrieve names
    ]

    # Separate dataset into features (X) and target (y)
    X = df[important_features].copy()  # Include 'Player' in X for reference
    y = df['Value'].copy()  # Target variable

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit and transform scaler on X (only numerical columns)
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Oversample each position category to balance the dataset
    df_defender = df[df['Position'] == 'Defender']
    df_midfield = df[df['Position'] == 'midfield']
    df_attack = df[df['Position'] == 'attack']
    df_goalkeeper = df[df['Position'] == 'Goalkeeper']

    # Determine the size of the majority class (Defender, in this case)
    majority_class_size = max(len(df_defender), len(df_midfield), len(df_attack), len(df_goalkeeper))

    # Oversample each minority class to match majority class size
    df_defender_oversampled = resample(df_defender,
                                       replace=True,  # Sample with replacement to replicate data
                                       n_samples=majority_class_size,  # Match majority class size
                                       random_state=42)  # Set random state for reproducibility

    df_midfield_oversampled = resample(df_midfield,
                                       replace=True,
                                       n_samples=majority_class_size,
                                       random_state=42)

    df_attack_oversampled = resample(df_attack,
                                     replace=True,
                                     n_samples=majority_class_size,
                                     random_state=42)

    df_goalkeeper_oversampled = resample(df_goalkeeper,
                                         replace=True,
                                         n_samples=majority_class_size,
                                         random_state=42)

    # Combine oversampled dataframes for all positions
    df_balanced = pd.concat([df_defender_oversampled, df_midfield_oversampled, df_attack_oversampled, df_goalkeeper_oversampled])

    # Shuffle the dataset to ensure randomness
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split balanced dataset into features (X_balanced) and target (y_balanced)
    X_balanced = df_balanced[important_features].copy()  # Include 'Player' in X_balanced for reference
    y_balanced = df_balanced['Value'].copy()  # Target variable

    # Drop 'Player' from X_balanced as it's not needed for prediction
    X_balanced.drop(columns=['Player'], inplace=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Initialize GradientBoostingRegressor with parameters
    gb_model = GradientBoostingRegressor(
        learning_rate=0.031646031047461894,
        max_depth=4,
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=400,
        subsample=0.8050678774612785,
        random_state=42
    )

    # Fit the model on the training data
    gb_model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred_gb = gb_model.predict(X_test)

    # Calculate undervaluation (predicted - actual)
    undervaluation = y_pred_gb - y_test.values

    # Calculate overvaluation (actual - predicted)
    overvaluation = y_test.values - y_pred_gb

    # Create a DataFrame to analyze undervaluation and overvaluation
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred_gb,
        'Undervaluation': undervaluation,
        'Overvaluation': overvaluation,
        'Player': df_balanced.loc[X_test.index, 'Player'].values  # Retrieve player names from original dataset
    })

    # Sort by undervaluation to find undervalued players
    undervalued_players = results_df.sort_values(by='Undervaluation', ascending=False)

    # Remove duplicates due to oversampling
    undervalued_players = undervalued_players.drop_duplicates(subset=['Player'], keep='first')

    # Sort by overvaluation to find overvalued players
    overvalued_players = results_df.sort_values(by='Overvaluation', ascending=False)

    # Remove duplicates due to oversampling
    overvalued_players = overvalued_players.drop_duplicates(subset=['Player'], keep='first')

    # Prepare the result to be returned by the API
    overvalued = overvalued_players.head(25).to_dict(orient='records')
    undervalued = undervalued_players.head(25).to_dict(orient='records')

    return {"overvalued": overvalued, "undervalued": undervalued}


@app.post("/predict")
async def predict(player_data: PlayerData):
    # Load your CSV file
    csv_file = '/Users/abhinavgoel/Downloads/final_dataset.csv'
    df = pd.read_csv(csv_file)

    # Drop duplicates based on 'Player'
    df = df.drop_duplicates(subset=['Player'], keep='first')

    # Define position weights based on value hierarchy
    position_map = {
        'attack': 4,
        'midfield': 3,
        'Defender': 2,
        'Goalkeeper': 1
    }
    df['Position_Value'] = df['Position'].map(position_map)

    # Map "League" column
    league_map = {
        'Premier League': 5,
        'Bundesliga': 4,
        'Serie A': 2,
        'La Liga': 3,
        'Ligue 1': 1
    }
    df['League'] = df['League'].map(league_map)

    # Drop unwanted columns
    unwanted_columns = ['Club', 'Nation']  # Keep 'Player' for now
    df.drop(columns=unwanted_columns, inplace=True, errors='ignore')

    # Define a function to apply age and contract year weights
    def apply_age_contract_weights(df):
        # Weightage for contract years left
        df['Contract_Weight'] = df['Contract Years Left'].apply(lambda years_left: 3 if years_left >= 5 else (2 if years_left >= 2 else 1))
        return df

    # Apply age and contract weights
    df = apply_age_contract_weights(df)

    # Define weight mapping for different seasons
    season_weights = {
        '(20/21)': 1.0,
        '(19/20)': 0.93,
        '(18/19)': 0.75,
        '(17/18)': 0.58
    }

    # Function to apply season weights
    def apply_season_weights(df, season_weights):
        for season, weight in season_weights.items():
            season_cols = [col for col in df.columns if season in col]
            for col in season_cols:
                df[col] *= weight
        return df

    # Apply season weights
    df = apply_season_weights(df, season_weights)

    # List of most important features
    important_features = [
        'Passes Leading to Goals (20/21)',
        'Age',
        'Number of Times Player was Pass Target (20/21)',
        'Number of Times Received Pass (19/20)',
        'Touches in Attacking Penalty Box (20/21)',
        'Total Carries in Forward Direction (20/21)',
        'Contract Years Left',
        'Successful Pressure % (17/18)',
        'Carries into Attacking Penalty Box (20/21)',
        'Goal Creating Actions/90 (19/20)',
        '(Gls+Ast-Scored Penalties)/90 (19/20)',
        'Goal Creating Actions (20/21)',
        '(G+A)/90 (19/20)',
        '(G+A)/90 (20/21)',
        'xG (20/21)',
        'Number of Times Player was Pass Target (18/19)',
        'Number of Times Received Pass (20/21)',
        'League',
        'Number of Times Player was Pass Target (19/20)',
        'Position_Value',
        'Player'  # Add 'Player' column back to retrieve names
    ]

    # Separate dataset into features (X) and target (y)
    X = df[important_features].copy()  # Include 'Player' in X for reference
    y = df['Value'].copy()  # Target variable

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit and transform scaler on X (only numerical columns)
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Oversample each position category to balance the dataset
    df_defender = df[df['Position'] == 'Defender']
    df_midfield = df[df['Position'] == 'midfield']
    df_attack = df[df['Position'] == 'attack']
    df_goalkeeper = df[df['Position'] == 'Goalkeeper']

    # Determine the size of the majority class (Defender, in this case)
    majority_class_size = max(len(df_defender), len(df_midfield), len(df_attack), len(df_goalkeeper))

    # Oversample each minority class to match majority class size
    df_defender_oversampled = resample(df_defender,
                                       replace=True,  # Sample with replacement to replicate data
                                       n_samples=majority_class_size,  # Match majority class size
                                       random_state=42)  # Set random state for reproducibility

    df_midfield_oversampled = resample(df_midfield,
                                       replace=True,
                                       n_samples=majority_class_size,
                                       random_state=42)

    df_attack_oversampled = resample(df_attack,
                                     replace=True,
                                     n_samples=majority_class_size,
                                     random_state=42)

    df_goalkeeper_oversampled = resample(df_goalkeeper,
                                         replace=True,
                                         n_samples=majority_class_size,
                                         random_state=42)

    # Combine oversampled dataframes for all positions
    df_balanced = pd.concat([df_defender_oversampled, df_midfield_oversampled, df_attack_oversampled, df_goalkeeper_oversampled])

    # Shuffle the dataset to ensure randomness
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split balanced dataset into features (X_balanced) and target (y_balanced)
    X_balanced = df_balanced[important_features].copy()  # Include 'Player' in X_balanced for reference
    y_balanced = df_balanced['Value'].copy()  # Target variable

    # Drop 'Player' from X_balanced as it's not needed for prediction
    X_balanced.drop(columns=['Player'], inplace=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Initialize GradientBoostingRegressor with parameters
    gb_model = GradientBoostingRegressor(
        learning_rate=0.031646031047461894,
        max_depth=4,
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=400,
        subsample=0.8050678774612785,
        random_state=42
    )

    # Fit the model on the training data
    gb_model.fit(X_train, y_train)

    data = [[
        player_data.PassesLeadingToGoals,
        player_data.Age,
        player_data.NumberOfTimesPassTarget,
        player_data.NumberOfTimesReceivedPass,
        player_data.TouchesInAttackingPenaltyBox,
        player_data.TotalCarriesInForwardDirection,
        player_data.ContractYearsLeft,
        player_data.SuccessfulPressurePercent,
        player_data.CarriesIntoAttackingPenaltyBox,
        player_data.GoalCreatingActionsPer90,
        player_data.GlsAstScoredPenaltiesPer90,
        player_data.GoalCreatingActions,
        player_data.GA90,
        player_data.xG,
        player_data.League,
        player_data.Position,
    ]]
    prediction = gb_model.predict(data)
    is_overvalued = prediction[0] > player_data.Value
    return {
        "predictedValue": prediction[0],
        "isOvervalued": is_overvalued
    }
