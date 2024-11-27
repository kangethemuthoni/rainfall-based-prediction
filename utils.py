import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

country_categories = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 
    'Central African Republic', 'Chad', 'Comoros', 'Congo (Congo-Brazzaville)', 'Congo (Democratic Republic)', 
    'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 
    'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 
    'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 
    'Sao Tome and Principe', 'Senegal', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 
    'Uganda', 'Zambia', 'Zimbabwe'
]

precomputed_rainfall = {
    ("Kenya", 1): 0.1,
    ("Kenya", 6): 0.2,
    ("Malawi", 1): 0.3,
    ("Sudan", 1): 0.6,
    ("Algeria", 1): 0.05,  
    ("Angola", 1): 0.1,    
    ("Benin", 1): 0.15,   
    ("Botswana", 1): 0.05, 
    ("Burkina Faso", 1): 0.2,  
    ("Burundi", 1): 0.25,  
    ("Cabo Verde", 1): 0.1,  
    ("Cameroon", 1): 0.3,  
    ("Central African Republic", 1): 0.2,  
    ("Chad", 1): 0.05,  
    ("Comoros", 1): 0.4,  
    ("Congo (Congo-Brazzaville)", 1): 0.3,  
    ("Congo (Democratic Republic)", 1): 0.5,  
    ("Djibouti", 1): 0.02,  
    ("Egypt", 1): 0.01, 
    ("Equatorial Guinea", 1): 0.3,  
    ("Eritrea", 1): 0.05,  
    ("Eswatini", 1): 0.2,  
    ("Ethiopia", 1): 0.3,  
    ("Gabon", 1): 0.4,  
    ("Gambia", 1): 0.1,  
    ("Ghana", 1): 0.2,  
    ("Guinea", 1): 0.3,  
    ("Guinea-Bissau", 1): 0.3,  
    ("Ivory Coast", 1): 0.2,  
    ("Lesotho", 1): 0.1,  
    ("Liberia", 1): 0.3,  
    ("Libya", 1): 0.01,  
    ("Madagascar", 1): 0.4,  
    ("Mali", 1): 0.05,  
    ("Mauritania", 1): 0.02,  
    ("Mauritius", 1): 0.3,  
    ("Morocco", 1): 0.05,  
    ("Mozambique", 1): 0.3,  
    ("Namibia", 1): 0.02,  
    ("Niger", 1): 0.05, 
    ("Nigeria", 1): 0.2,  
    ("Rwanda", 1): 0.3,  
    ("Sao Tome and Principe", 1): 0.4,  
    ("Senegal", 1): 0.25,  
    ("Somalia", 1): 0.15,  
    ("South Africa", 1): 0.3,  
    ("South Sudan", 1): 0.3,  
    ("Tanzania", 1): 0.3, 
    ("Togo", 1): 0.2,  
    ("Tunisia", 1): 0.05,  
    ("Uganda", 1): 0.3,  
    ("Zambia", 1): 0.3,  
    ("Zimbabwe", 1): 0.3,  
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