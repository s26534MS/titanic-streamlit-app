import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Wczytanie danych
base_data = pd.read_csv("DSP_1.csv")

# Wybór kolumn
cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
data = base_data[cols].copy()

# Uzupełnienie brakujących wartości wieku (usuwa FutureWarning)
data.loc[:, "Age"] = data["Age"].fillna(data["Age"].mean())

# Usunięcie braków w 'Embarked'
data.dropna(subset=['Embarked'], inplace=True)

# Zakodowanie zmiennych kategorycznych
encoder = LabelEncoder()
data.loc[:, "Sex"] = encoder.fit_transform(data.loc[:, "Sex"])
data.loc[:, "Embarked"] = encoder.fit_transform(data.loc[:, "Embarked"])

# Podział na cechy (X) i etykiety (y)
y = data.iloc[:, 0]
X = data.iloc[:, 1:8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Funkcja trenująca model
def train_model(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=20, random_state=0)
    forest.fit(X_train, y_train)
    print("Dokładność modelu:", forest.score(X_train, y_train))
    return forest


# Trenujemy model
forest = train_model(X_train, y_train)

# Przykładowe dane testowe (poprawiony format)
my_data = pd.DataFrame([
    [1, 1, 50, 0, 0, 0, 2]
], columns=X.columns)

# Predykcja (usuwa UserWarning)
print(forest.predict(my_data))

# Kolejna próbka
my_data = np.array([
    [1, 0, 20, 1, 0, 0, 2]
])

# Predykcja
print(forest.predict(my_data))

# Zapis modelu
filename = "../materiały/model.sv"
pickle.dump(forest, open(filename, "wb"))
