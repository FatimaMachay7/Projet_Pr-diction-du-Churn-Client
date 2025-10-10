
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  accuracy_score,recall_score,precision_score,f1_score,roc_curve, auc
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    df= pd.read_csv("./DATA/churn.csv")
    return df
data= load_data()
print(data)


def clean_data(data1):
    colonnes_supprimer = ['customerID', 'gender', 'Partner', 'SeniorCitizen']
    data_cleaned = data1.drop(columns=colonnes_supprimer, errors='ignore')
    data_cleaned['TotalCharges'] = pd.to_numeric(data_cleaned['TotalCharges'], errors='coerce')
    data_cleaned['TotalCharges'] = data_cleaned['TotalCharges'].fillna(data_cleaned['TotalCharges'].mean())
    return data_cleaned
data_cleaned=clean_data(data)
print(data_cleaned)


def encodage_cat(data_cleaned):
    encoder = LabelEncoder()
    data_cleaned['Churn'] = encoder.fit_transform(data['Churn'])
    columns_to_encode = ['Dependents', 'PhoneService', 'MultipleLines', 
                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 
                         'StreamingMovies', 'Contract', 'PaperlessBilling', 
                         'PaymentMethod']
    for col in columns_to_encode:
        if col in data_cleaned.columns:
            encoder = LabelEncoder()
            data_cleaned[col] = encoder.fit_transform(data_cleaned[col])
            print(f"Colonne encodée: {col}")
        else:
            print(f"Attention : la colonne {col} n'existe pas dans le dataframe.")
    return data_cleaned
data_encoded = encodage_cat(data_cleaned)
print(data_encoded.head())



def separation(data_cleaned):
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in ['Churn', 'SeniorCitizen']]
    print("Colonnes numériques après exclusion :", numeric_columns)
    columns_to_encode = ['Dependents', 'PhoneService', 'MultipleLines', 
                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 
                         'StreamingMovies', 'Contract', 'PaperlessBilling', 
                         'PaymentMethod']
    encoded_columns = [col for col in columns_to_encode if col in data_cleaned.columns]
    final_data =pd.concat([data_cleaned[numeric_columns], data_cleaned[encoded_columns], data_cleaned[['Churn']]],axis=1)
    return final_data
final_data =separation(data_cleaned)
print(final_data.columns,"final_data.columns")



def separation_x_y(d_c):
    Y = d_c['Churn']
    X = d_c.drop('Churn', axis=1)
    return X, Y
X, Y = separation_x_y(final_data)
print(X.head(5))
print(Y.head(5))



def split_train_test(X,Y):
    X_tr, X_ts,Y_Tr,Y_ts = train_test_split(X, Y,test_size=0.2, random_state=42)
    return X_tr, X_ts,Y_Tr,Y_ts

X_tr, X_ts,Y_Tr,Y_ts= split_train_test (X,Y)
print("split ", X_tr)



def normalisation(X1,X2):
    scaler = MinMaxScaler()
    X_TR_S = scaler.fit_transform(X1)
    X_ts_S = scaler.transform(X2)
    return X_TR_S,X_ts_S
X_TR_S,X_ts_S=normalisation(X_tr,X_ts)
print("voila", X_TR_S)
print("Forme X_TR_S:", X_TR_S.shape)
print("Extrait X_TR_S:", X_TR_S[:5])



def RandomForest_Classifier(x, y, X):
    model = RandomForestClassifier(random_state=42)
    model.fit(x, y)
    y_pred = model.predict(X)
    print("Prédictions :", y_pred)
    y_proba = model.predict_proba(X)[:, 1]
    print("Probabilités (classe 1) :", y_proba)
    return model, y_pred, y_proba
model, y_pred, y_proba = RandomForest_Classifier(X_TR_S, Y_Tr, X_ts_S)
print("voila données",model, y_pred, y_proba)



def metriques_randomForest_Classifier(Y_ts,y_pred):
    accuracy_RC = accuracy_score(Y_ts,y_pred)
    recall_RC =recall_score(Y_ts,y_pred)
    precision_RC = precision_score(Y_ts,y_pred)
    f1_RC = f1_score(Y_ts,y_pred)
    return accuracy_RC,recall_RC,precision_RC,f1_RC
accuracy_RC,recall_RC,precision_RC,f1_RC=metriques_randomForest_Classifier(Y_ts,y_pred)
print("Accuracy :", round(accuracy_RC, 4))
print("Recall :", round(recall_RC, 4))
print("Precision :", round(precision_RC, 4))
print("F1 Score :", round(f1_RC, 4))



def plot_roc_curve(Y_ts, y_proba, model_name='Modèle'):
    fpr, tpr, thresholds = roc_curve(Y_ts, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonale (aléatoire)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    return fpr, tpr, thresholds
plot_roc_curve(Y_ts, y_proba, model_name="Random Forest")




def  Logistic_Regression(X, x, Y):
    model1 = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)
    model1.fit(X, Y)
    y_predict= model.predict(x)
    y_proba = model.predict_proba(x)[:,1]
    print(y_proba)
    y_predict = model.predict(x)
    return model1,y_predict,y_proba 
model1,y_predict,y_proba=Logistic_Regression(X_TR_S, X_ts_S, Y_Tr)
print('La voila :', model1,y_predict,y_proba)



def metriques_Logistic_Regression(Y_ts,y_pred):
    accuracy_RL = accuracy_score(Y_ts,y_pred)
    recall_RL =recall_score(Y_ts,y_pred)
    precision_RL = precision_score(Y_ts,y_pred)
    f1_RL = f1_score(Y_ts,y_pred)
    return accuracy_RL,recall_RL,precision_RL,f1_RL
accuracy_RL,recall_RL,precision_RL,f1_RL=metriques_Logistic_Regression(Y_ts,y_pred)
print("Accuracy :", round(accuracy_RL, 4))
print("Recall :", round(recall_RL, 4))
print("Precision :", round(precision_RL, 4))
print("F1 Score :", round(f1_RL, 4))



def plot_roc_curve_RL(Y_ts, y_proba, model_name='Modèle'):
    fpr, tpr, thresholds = roc_curve(Y_ts, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    return fpr, tpr, thresholds
plot_roc_curve_RL(Y_ts, y_proba, model_name="Logistic Regression")



def M_SVC(X, x, Y):
    model1 = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42)
    model1.fit(X, Y)
    y_predict= model.predict(x)
    y_proba = model.predict_proba(x)[:,1]
    print(y_proba)
    y_predict = model.predict(x)
    return model1,y_predict,y_proba 
model1,y_predict,y_proba=M_SVC(X_TR_S, X_ts_S, Y_Tr)
print('La voila :', model1,y_predict,y_proba)


def metriques_SVC(Y_ts,y_pred):
    accuracy_S = accuracy_score(Y_ts,y_pred)
    recall_S =recall_score(Y_ts,y_pred)
    precision_S = precision_score(Y_ts,y_pred)
    f1_S = f1_score(Y_ts,y_pred)
    return accuracy_S,recall_S,precision_S,f1_S
accuracy_S,recall_S,precision_S,f1_S=metriques_SVC(Y_ts,y_pred)
print("Accuracy :", round(accuracy_S, 4))
print("Recall :", round(recall_S, 4))
print("Precision :", round(precision_S, 4))
print("F1 Score :", round(f1_S, 4))



def plot_roc_curve_SVC(Y_ts, y_proba, model_name='Modèle'):
    fpr, tpr, thresholds = roc_curve(Y_ts, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='green', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='-')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    return fpr, tpr, thresholds
plot_roc_curve_SVC(Y_ts, y_proba, model_name="SVC")