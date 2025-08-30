import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from google.colab import files
uploaded = files.upload()

def preprocess_data(df):
    """Preprocess the hurricane data with robust coordinate parsing"""
    data = df.copy()

    # Convert date and time to datetime
    data['DateTime'] = pd.to_datetime(
        data['Date'].astype(str) + ' ' + data['Time'].astype(str).str.zfill(4),
        format='%Y%m%d %H%M',
        errors='coerce'
    )

    def parse_coordinate(coord_str, coord_type='lat'):
        """Parse coordinate string handling various formats"""
        if pd.isna(coord_str):
            return np.nan

        coord_str = str(coord_str).strip().strip('-')
        if not coord_str or coord_str == '-999':
            return np.nan

        if coord_type == 'lat':
            multiplier = -1 if 'S' in coord_str else 1
            numeric_part = coord_str.replace('S', '').replace('N', '').strip()
        else:
            multiplier = -1 if 'W' in coord_str else 1
            numeric_part = coord_str.replace('W', '').replace('E', '').strip()

        numeric_part = ''.join(c for c in numeric_part if c.isdigit() or c == '.')

        try:
            return float(numeric_part) * multiplier
        except:
            return np.nan

    # Parse latitude and longitude
    data['Lat'] = data['Latitude'].apply(lambda x: parse_coordinate(x, 'lat'))
    data['Lon'] = data['Longitude'].apply(lambda x: parse_coordinate(x, 'lon'))

    # Handle missing values in key numeric columns
    numeric_columns = ['Max_Sustained_Wind_kt', 'Min_Pressure_mb', 'Radius_of_Max_Wind_nm']
    for col in numeric_columns:
        data[col] = data[col].replace(-999, np.nan)

    # Create storm identifier
    data['Storm_ID'] = data['Basin'] + '_' + data['Cyclone_Number'].astype(str) + '_' + data['Year'].astype(str)

    # Drop rows with invalid coordinates
    initial_len = len(data)
    data = data.dropna(subset=['Lat', 'Lon'])
    removed_rows = initial_len - len(data)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with invalid coordinates")

    # Sort by Storm and DateTime
    data = data.sort_values(['Storm_ID', 'DateTime'])

    return data

df = pd.read_csv('/content/data.csv')  # Make sure the name matches

# Preprocess the data
processed_data = preprocess_data(df)

# Show the first few rows
processed_data.sample(10)

processed_data.head(5)

def create_flexible_sequences(df, min_sequence_length=4, max_sequence_length=50, forecast_horizon=4):
    """
    Create training sequences of variable length and corresponding future targets.

    Args:
        df: Preprocessed hurricane DataFrame.
        min_sequence_length: Minimum number of past observations.
        max_sequence_length: Maximum number of past observations to use.
        forecast_horizon: How many 6-hour steps ahead to forecast.

    Returns:
        A list of dictionaries with input sequences and targets.
    """
    sequences = []

    for storm_id, group in df.groupby('Storm_ID'):
        group = group.reset_index(drop=True)
        storm_length = len(group)

        # Skip storms that are too short
        if storm_length < min_sequence_length + 1:
            continue

        for i in range(storm_length - min_sequence_length):
            available_past = i + 1
            sequence_length = min(available_past, max_sequence_length)

            if sequence_length < min_sequence_length:
                continue

            # Input sequence: the past `sequence_length` observations
            start_idx = max(0, i + 1 - sequence_length)
            input_seq = group.iloc[start_idx:i + 1]

            # Targets: lat/lon for future steps (up to forecast_horizon)
            target_positions = []
            for j in range(1, min(forecast_horizon + 1, storm_length - i)):
                target_idx = i + j
                if target_idx < storm_length:
                    target_positions.append({
                        'lat': group.iloc[target_idx]['Lat'],
                        'lon': group.iloc[target_idx]['Lon'],
                        'step': j
                    })

            if target_positions:
                sequences.append({
                    'storm_id': storm_id,
                    'input_sequence': input_seq,
                    'targets': target_positions,
                    'sequence_length': len(input_seq)
                })

    return sequences

# Set the desired parameters
MIN_SEQ_LEN = 4
MAX_SEQ_LEN = 50
FORECAST_HORIZON = 4

# Generate sequences
sequences = create_flexible_sequences(
    processed_data,
    min_sequence_length=MIN_SEQ_LEN,
    max_sequence_length=MAX_SEQ_LEN,
    forecast_horizon=FORECAST_HORIZON
)

# Show a sample sequence
print(f"Total sequences created: {len(sequences)}\n")
print("Sample sequence (index 0):")
sample = sequences[0]

print(f"Storm ID: {sample['storm_id']}")
print(f"Input sequence length: {sample['sequence_length']}")
print("Input sequence columns:", list(sample['input_sequence'].columns))
print("Target steps:", sample['targets'])

import random

# Ensure reproducibility
random.seed(42)

# Select 5 random sequences
sample_sequences = random.sample(sequences, min(5, len(sequences)))

# Display each sequence
for idx, seq in enumerate(sample_sequences):
    print("="*60)
    print(f"🌀 Sequence {idx + 1}")
    print(f"Storm ID: {seq['storm_id']}")
    print(f"Input Sequence Length: {seq['sequence_length']}")

    print("\n📌 Input Observations:")
    display_columns = ['DateTime', 'Lat', 'Lon', 'Max_Sustained_Wind_kt', 'Min_Pressure_mb']
    print(seq['input_sequence'][display_columns])

    print("\n🎯 Targets (Future Predictions):")
    for target in seq['targets']:
        print(f"  +{target['step']*6}h → Lat: {target['lat']:.2f}, Lon: {target['lon']:.2f}")

import matplotlib.pyplot as plt

def plot_hurricane_sequence(seq, title="Hurricane Track Sequence"):
    input_seq = seq['input_sequence']
    targets = seq['targets']

    input_lats = input_seq['Lat'].values
    input_lons = input_seq['Lon'].values

    target_lats = [t['lat'] for t in targets]
    target_lons = [t['lon'] for t in targets]
    forecast_hours = [t['step'] * 6 for t in targets]

    plt.figure(figsize=(10, 6))

    # Plot input sequence
    plt.plot(input_lons, input_lats, 'bo-', label='Input Track (Past)', markersize=6)

    # Plot future predictions (targets)
    plt.plot(target_lons, target_lats, 'ro-', label='Target Track (Future)', markersize=6)

    # Annotate forecast steps
    for i, (lon, lat, hour) in enumerate(zip(target_lons, target_lats, forecast_hours)):
        plt.annotate(f'+{hour}h', (lon, lat), textcoords='offset points', xytext=(5, 5))

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot 5 random sequences
sample_sequences = random.sample(sequences, min(5, len(sequences)))

for idx, seq in enumerate(sample_sequences):
    title = f"🌀 Sequence {idx+1} — Storm: {seq['storm_id']}, Length: {seq['sequence_length']}"
    plot_hurricane_sequence(seq, title=title)

from sklearn.preprocessing import LabelEncoder

# Create and fit encoder
basin_encoder = LabelEncoder()
basin_encoder.fit(processed_data['Basin'].unique())

def extract_features(sequence, basin_encoder, max_sequence_length=6):
    features = []
    seq_len = len(sequence)

    # Pad/truncate observations up to max_sequence_length
    for i in range(max_sequence_length):
        if i < seq_len:
            obs = sequence.iloc[i]

            # Positional features
            features.extend([obs['Lat'], obs['Lon']])

            # Intensity features with fallback
            features.extend([
                obs['Max_Sustained_Wind_kt'] if pd.notna(obs['Max_Sustained_Wind_kt']) else 50,
                obs['Min_Pressure_mb'] if pd.notna(obs['Min_Pressure_mb']) else 1000,
                obs['Radius_of_Max_Wind_nm'] if pd.notna(obs['Radius_of_Max_Wind_nm']) else 30
            ])

            # Time features
            if pd.notna(obs['DateTime']):
                features.extend([
                    obs['DateTime'].hour,
                    obs['DateTime'].dayofyear,
                    obs['DateTime'].month
                ])
            else:
                features.extend([12, 200, 7])
        else:
            # Padding for missing observations (8 features per step)
            features.extend([0] * 8)

    # Motion features (velocity, speed, direction)
    if seq_len >= 2:
        lat_vel = sequence.iloc[-1]['Lat'] - sequence.iloc[-2]['Lat']
        lon_vel = sequence.iloc[-1]['Lon'] - sequence.iloc[-2]['Lon']
        speed = np.sqrt(lat_vel**2 + lon_vel**2)
        direction = np.arctan2(lat_vel, lon_vel)
        features.extend([lat_vel, lon_vel, speed, direction])

        # Acceleration
        if seq_len >= 3:
            prev_lat_vel = sequence.iloc[-2]['Lat'] - sequence.iloc[-3]['Lat']
            prev_lon_vel = sequence.iloc[-2]['Lon'] - sequence.iloc[-3]['Lon']
            lat_acc = lat_vel - prev_lat_vel
            lon_acc = lon_vel - prev_lon_vel
            features.extend([lat_acc, lon_acc])
        else:
            features.extend([0, 0])
    else:
        features.extend([0, 0, 0, 0, 0, 0])  # Padding if not enough data

    # Add basin and sequence length
    basin_encoded = basin_encoder.transform([sequence.iloc[0]['Basin']])[0]
    features.extend([basin_encoded, seq_len])

    return features

# Try feature extraction on the first sequence
features = extract_features(sequences[0]['input_sequence'], basin_encoder, max_sequence_length=6)

print(f"Feature vector length: {len(features)}")
print("First 10 features:", features[:10])

# Randomly sample 10 sequences
sample_sequences = random.sample(sequences, min(10, len(sequences)))

print("="*70)
print("🔍 Showing extracted features for 10 random sequences\n")

for idx, seq in enumerate(sample_sequences):
    print("="*70)
    print(f"🌀 Sequence {idx+1}")
    print(f"Storm ID: {seq['storm_id']}")
    print(f"Input Sequence Length: {seq['sequence_length']}")

    # Extract features
    feat = extract_features(seq['input_sequence'], basin_encoder, max_sequence_length=6)

    # Display features
    print(f"\n🧩 Feature Vector Length: {len(feat)}")
    print(f"📋 First 10 Features: {feat[:10]}")
    print(f"📋 Last 5 Features (Velocity/Accel/Basin/Len): {feat[-7:]}")

def prepare_training_data(sequences, basin_encoder, max_sequence_length=6):
    X = []
    y_lat = []
    y_lon = []

    for seq_data in sequences:
        features = extract_features(seq_data['input_sequence'], basin_encoder, max_sequence_length)

        for target in seq_data['targets']:
            # Append forecast step to the end
            X.append(features + [target['step']])
            y_lat.append(target['lat'])
            y_lon.append(target['lon'])

    return np.array(X), np.array(y_lat), np.array(y_lon)

from sklearn.model_selection import train_test_split

# Get all unique storm IDs
storm_ids = list(set([seq['storm_id'] for seq in sequences]))

# Train/test split by storm ID
train_storm_ids, test_storm_ids = train_test_split(
    storm_ids, test_size=0.2, random_state=42
)

# Separate sequences
train_sequences = [seq for seq in sequences if seq['storm_id'] in train_storm_ids]
test_sequences = [seq for seq in sequences if seq['storm_id'] in test_storm_ids]

print(f"✅ Training storms: {len(train_storm_ids)}, Testing storms: {len(test_storm_ids)}")
print(f"✅ Training sequences: {len(train_sequences)}, Testing sequences: {len(test_sequences)}")

# Build datasets
X_train, y_lat_train, y_lon_train = prepare_training_data(train_sequences, basin_encoder, max_sequence_length=6)
X_test, y_lat_test, y_lon_test = prepare_training_data(test_sequences, basin_encoder, max_sequence_length=6)

print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"🧩 Feature dimension: {X_train.shape[1]}")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

# Initialize models
lat_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=30,n_jobs=-1)
lon_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=30,n_jobs=-1)

# Train
print("🎯 Training latitude model...")
lat_model.fit(X_train_scaled, y_lat_train)

print("🎯 Training longitude model...")
lon_model.fit(X_train_scaled, y_lon_train)

from sklearn.metrics import mean_absolute_error
import numpy as np

# Predict
lat_pred = lat_model.predict(X_test_scaled)
lon_pred = lon_model.predict(X_test_scaled)

# Evaluate
lat_mae = mean_absolute_error(y_lat_test, lat_pred)
lon_mae = mean_absolute_error(y_lon_test, lon_pred)

# Distance error in nautical miles
distance_error = np.sqrt((lat_pred - y_lat_test)**2 + (lon_pred - y_lon_test)**2) * 60
avg_distance_error = np.mean(distance_error)

# Print results
print("\n📈 Model Performance:")
print(f"Latitude MAE:  {lat_mae:.4f}°")
print(f"Longitude MAE: {lon_mae:.4f}°")
print(f"🌊 Avg Approximate Distance Error: {avg_distance_error:.2f} nautical miles")

from datetime import timedelta

def predict_track(input_sequence, lat_model, lon_model, scaler, basin_encoder,
                  forecast_steps=4, max_sequence_length=50):
    predictions = []

    for step in range(1, forecast_steps + 1):
        features = extract_features(input_sequence, basin_encoder, max_sequence_length)
        features.append(step)  # Add step to features
        X_scaled = scaler.transform([features])

        # Predict
        pred_lat = lat_model.predict(X_scaled)[0]
        pred_lon = lon_model.predict(X_scaled)[0]

        # Predict time
        last_time = input_sequence.iloc[-1]['DateTime']
        pred_time = last_time + timedelta(hours=6 * step) if pd.notna(last_time) else None

        predictions.append({
            'step': step,
            'forecast_hour': step * 6,
            'predicted_lat': pred_lat,
            'predicted_lon': pred_lon,
            'predicted_time': pred_time
        })

    return predictions

def plot_track_prediction(historical_data, predictions, title="Hurricane Track Prediction"):
    plt.figure(figsize=(10, 6))

    # Historical path
    plt.plot(historical_data['Lon'], historical_data['Lat'], 'bo-', label='Historical Track')

    # Predicted path
    pred_lons = [p['predicted_lon'] for p in predictions]
    pred_lats = [p['predicted_lat'] for p in predictions]
    plt.plot(pred_lons, pred_lats, 'ro-', label='Predicted Track')

    # Labels
    for p in predictions:
        plt.annotate(f"+{p['forecast_hour']}h", (p['predicted_lon'], p['predicted_lat']),
                     textcoords="offset points", xytext=(5, 5))

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Group processed data by storm
storm_groups = processed_data.groupby('Storm_ID')
train_storm_ids_set = set(train_storm_ids)

# Get training storms only
train_storm_groups = [(sid, group) for sid, group in storm_groups if sid in train_storm_ids_set]

# Pick 10 random storms
random.seed()
sample_storms = random.sample(train_storm_groups, min(10, len(train_storm_groups)))

# Loop through and predict
for idx, (storm_id, storm_data) in enumerate(sample_storms):
    storm_data = storm_data.sort_values('DateTime')
    input_data = storm_data.iloc[:min(6, len(storm_data))]  # Use up to 6 obs

    predictions = predict_track(
        input_sequence=input_data,
        lat_model=lat_model,
        lon_model=lon_model,
        scaler=scaler,
        basin_encoder=basin_encoder,
        forecast_steps=4,
        max_sequence_length=6
    )

    print("=" * 70)
    print(f"🌀 Storm {idx+1}: {storm_id}")
    print(f"Used {len(input_data)} observations to predict next 24 hours (4 x 6h)\n")

    for pred in predictions:
        print(f"  +{pred['forecast_hour']}h → Lat: {pred['predicted_lat']:.2f}, Lon: {pred['predicted_lon']:.2f}")

    plot_track_prediction(storm_data.iloc[:min(10, len(storm_data))], predictions,
                          title=f"{storm_id} – Predicted Track")
