from pipeline import load_data

data = load_data()
print(data)
# data.drop('customerID', axis=1, inplace=True)