
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def load_data():
   data = pd.read_csv("./DATA/churn.csv")
   print(data)
   return data
load_data()

# def preparation() :
#     columns_to_encode = ['Dependents', 'PhoneService', 'MultipleLines', 
#                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
#                      'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
#                      'PaperlessBilling', 'PaymentMethod']
#     for col in columns_to_encode: 
#         encoder=LabelEncoder()
#     data[col] = encoder.fit_transform(data[col])
#     print(f"Colonne encodée: {col}")
#     print(data[col].value_counts())
#     print("-" * 50)
