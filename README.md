# 🚨 **Fraud Detection System** 🚨

A machine learning-based fraud detection system built using Python, SQLite, and various classifiers like Random Forest, XGBoost, and Logistic Regression. This project uses real-time transaction data and predicts fraudulent transactions.


## 📊 **Project Overview**

This project aims to predict fraudulent credit card transactions by using an ensemble machine learning model. It utilizes:

- **Random Forest Classifier**
- **XGBoost Classifier**
- **Logistic Regression**

The model is trained on the Kaggle dataset for credit card transactions and uses **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance.


## 🛠️ **Technologies Used**

- **Python** 🐍
- **pandas** 📊
- **NumPy** 🔢
- **scikit-learn** 🔧
- **XGBoost** 🚀
- **SMOTE (imbalanced-learn)** ⚖️
- **SQLite** 🗄️
- **time** ⏱️


## 📥 **How to Run the Project**

1. **Clone the Repository**

   Clone the repository to your local machine using:

   ```
   git clone https://github.com/ProfPython/FraudDetectionModel.git
   
   ```
2. **Install Dependencies**

    Install the required Python packages listed in requirements.txt by running:

    ```
    pip install -r requirements.txt

    ```

3. **Download the DataSet**

    Download the dataset (creditcard.csv) from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the same directory as the script.

4. **Run the Script**

    Execute the Python script to train the model and simulate real-time fraud detection:

    ```
    python fraud_detection.py

    ```

## 🧑‍💻 **Code Explanation**

**Data Preprocessing**

*   The dataset (creditcard.csv) is loaded into a Pandas DataFrame.

*   The Amount feature is normalized using StandardScaler.

*   SMOTE is used to handle the class imbalance in the target variable (Class).

**Modelling**

*   Random Forest Classifier, XGBoost, and Logistic Regression models are used in a Voting Classifier to  predict fraudulent transactions.

*   The model is trained using the training set and evaluated using the testing set.

**Real-Time Prediction Simulation**

*   The script simulates real-time transaction processing where predictions are made one-by-one for each test sample.

*   Predictions are stored in an SQLite database (fraud_detection.db).

## Interactive Power BI Dashboard

This project also supports real-time fraud prediction visualization using Power BI. Follow these steps:

1.  Set up Power BI and connect it to the SQLite database.

2.  Create interactive visualizations, such as a pie chart of fraudulent vs legitimate transactions, and time series analysis of fraud over time.