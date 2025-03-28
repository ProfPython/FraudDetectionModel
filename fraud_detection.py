import pandas as pd
import numpy as np
import sqlite3
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Load the dataset 
data = pd.read_csv('creditcard.csv')

# Normalize the 'Amount' feature
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop(['Class', 'Time'], axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(random_state=42)

# Create a Voting Classifier to combine the models
ensemble_model = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('lr', lr_model)
], voting='hard')

ensemble_model.fit(X_train_res, y_train_res)

conn = sqlite3.connect('fraud_detection.db')

conn.execute('''
CREATE TABLE IF NOT EXISTS real_time_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER,
    amount REAL,
    prediction INTEGER,
    actual INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
''')
conn.commit()

# Simulate real-time transaction processing
def simulate_real_time(X_test, y_test, model, conn):
    print("Starting real-time fraud detection simulation...")
    
    # Reset index of X_test to make sure indices align for y_test
    X_test_reset = X_test.reset_index(drop=True)
    
    for index, row in X_test_reset.iterrows():
        # Get the corresponding actual class label from y_test
        actual_class = y_test.iloc[index]
        
        transaction = row.values.reshape(1, -1)
        prediction = model.predict(transaction)[0]
        
        conn.execute('''
        INSERT INTO real_time_predictions (transaction_id, amount, prediction, actual)
        VALUES (?, ?, ?, ?)
        ''', (index, row['Amount'], int(prediction), int(actual_class)))
        
        conn.commit()

        print(f"Transaction {index} - Predicted: {prediction} | Actual: {actual_class}")

        # Simulate a real-time delay (e.g., 1 second per transaction)
        time.sleep(1)

simulate_real_time(X_test.sample(100, random_state=42), y_test, ensemble_model, conn)

result = pd.read_sql_query('SELECT * FROM real_time_predictions LIMIT 10;', conn)
print(result)

conn.close()