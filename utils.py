import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Country categories for one-hot encoding
country_categories = ['KENYA', 'MALAWI', 'SUDAN']

# Precomputed rainfall data (example)
precomputed_rainfall = {
    ("KENYA", 1): 0.1,
    ("KENYA", 6): 0.2,
    ("MALAWI", 1): 0.3,
    ("SUDAN", 1): 0.6,
}

# Month mapping
month_mapping = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5, "JUNE": 6,
    "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12
}

# Initialize encoders and scalers
scaler = MinMaxScaler()
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
onehot_encoder.fit(np.array(country_categories).reshape(-1, 1))

def preprocess_month_input(month):
    if isinstance(month, str):
        month = month.upper()
        return month_mapping.get(month, -1)  # Return -1 if invalid
    return month

def process_disaster_record_for_model(record):
    """
    Processes a single record and prepares it for model input.
    """
    month = preprocess_month_input(record['Month'])
    country = record['Country']
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    disaster_int = 1  # Example disaster likelihood (Normal)
    country_features = onehot_encoder.transform([[country]]).flatten()

    feature_vector = [month_sin, month_cos, disaster_int]
    feature_vector.extend(country_features)

    # Pad to 78 features
    feature_vector.extend([0] * (78 - len(feature_vector)))

    return np.array(feature_vector).reshape(1, 1, -1)

def create_and_save_lstm_model():
    """
    Creates an LSTM model, compiles it, and saves it.
    """
    model = Sequential([
        LSTM(50, input_shape=(1, 78)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.save('models/mlp_model2.h5')
    print("LSTM model saved as 'models/lstm_model.h5'.")

if __name__ == "__main__":
    create_and_save_lstm_model()