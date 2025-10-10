import pytest

import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import *

df = pd.read_csv("./DATA/churn.csv")

@pytest.fixture
def prep_data():
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

def test_split_train_test(prep_data):
    X,y = prep_data
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Vérifier que X et y ont la même longueur
    assert len(X) == len(y)

 # Vérifier que le split conserve la correspondance lignes/étiquettes
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)




    