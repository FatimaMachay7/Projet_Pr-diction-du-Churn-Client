
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


def load_data():
   data_1 = pd.read_csv("./DATA/churn.csv")
   print(data_1)
   return data_1
load_data()

def preparation() :
   columns_to_encode = [, 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                     'PaperlessBilling', 'PaymentMethod']
   
columns_numeric=['tenure',]
tenure (ancienneté)

MonthlyCharges (charges mensuelles)

TotalCharges (charges totales)