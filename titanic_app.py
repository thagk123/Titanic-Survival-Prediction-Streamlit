import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Έλεγχος αν η εφαρμογή έχει ξεκινήσει
if "app_started" not in st.session_state:
    st.session_state["app_started"] = False

st.title("Titanic Survival Prediction")

# Κουμπί έναρξης
if not st.session_state["app_started"]:
    if st.button("🚀 Έναρξη Εφαρμογής"):
        st.session_state.clear()  # Καθαρίζει όλα τα session states
        st.session_state["app_started"] = True  # Ξεκινάει η εφαρμογή
        st.rerun()  # Επαναφόρτωση της σελίδας

# Αν η εφαρμογή ξεκινήσει, εμφανίζονται οι επιλογές
if st.session_state["app_started"]:

    # Φόρτωση δεδομένων
    @st.cache_data
    def load_data():
        df = pd.read_csv("train.csv")
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df.dropna(subset=['Embarked'], inplace=True)
        df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
        return df

    df = load_data()
    st.write("Προβολή δεδομένων:", df.head())

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # Αντικατάσταση κενών τιμών
    imputer = SimpleImputer(strategy="mean")
    X[:, 2:3] = imputer.fit_transform(X[:, 2:3])

    # Κανονικοποίηση
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Διαχωρισμός σε train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Επιλογή μοντέλου
    model_choice = st.selectbox("Επιλέξτε Μοντέλο", ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP"])

    # Αποθήκευση μοντέλου ως global για χρήση αργότερα
    model = None

    if st.button("Εκπαίδευση Μοντέλου"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=5)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        elif model_choice == "SVM":
            model = SVC(kernel='rbf', C=500, gamma=0.01)
        elif model_choice == "MLP":
            class MLP(nn.Module):
                def __init__(self):
                    super(MLP, self).__init__()
                    self.fc1 = nn.Linear(7, 10)
                    self.fc2 = nn.Linear(10, 1)

                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = torch.sigmoid(self.fc2(x))
                    return x

            model = MLP()
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

            train_data, train_labels = torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(1)
            for epoch in range(1000):
                optimizer.zero_grad()
                outputs = model(train_data)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()

        if model_choice != "MLP":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            test_data = torch.tensor(X_test).float()
            with torch.no_grad():
                y_pred = (model(test_data) >= 0.5).float().numpy().flatten()

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.session_state["model"] = model  # Αποθήκευση του μοντέλου στο session_state
        st.session_state["model_type"] = model_choice

        st.write(f"**Ακρίβεια:** {accuracy:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(cm)

    # Πρόβλεψη για νέο επιβάτη
    st.subheader("Δοκιμή με νέα δεδομένα")
    pclass = st.selectbox("Τάξη Εισιτηρίου", [1, 2, 3])
    sex = st.radio("Φύλο", ["Άνδρας", "Γυναίκα"])
    age = st.number_input("Ηλικία", 1, 100, 30)
    sibsp = st.number_input("Αδέρφια/Σύζυγοι", 0, 10, 0)
    parch = st.number_input("Γονείς/Παιδιά", 0, 10, 0)
    fare = st.number_input("Ναύλος", 0, 500, 50)
    embarked = st.selectbox("Λιμάνι Επιβίβασης", ["S", "C", "Q"])

    input_data = np.array([[pclass, 1 if sex == "Γυναίκα" else 0, age, sibsp, parch, fare, {"S": 0, "C": 1, "Q": 2}[embarked]]])
    input_data = scaler.transform(input_data)

    if st.button("Πρόβλεψη Επιβίωσης"):
        if "model" not in st.session_state:
            st.error("Παρακαλώ πρώτα εκπαιδεύστε ένα μοντέλο!")
        else:
            trained_model = st.session_state["model"]
            model_type = st.session_state["model_type"]

            if model_type != "MLP":
                prediction = trained_model.predict(input_data)[0]
            else:
                with torch.no_grad():
                    prediction = (trained_model(torch.tensor(input_data).float()) >= 0.5).item()

            st.write(f"**Πρόβλεψη:** {'Επιβίωσε' if prediction == 1 else 'Δεν Επιβίωσε'}")

    # Προσθήκη κουμπιού για επαναφορά εφαρμογής (χωρίς τερματισμό)
    if st.button("🔄 Επαναφορά Εφαρμογής"):
        st.session_state.clear()
        st.session_state["app_started"] = False
        st.rerun()

    # Προσθήκη κουμπιού για πλήρη τερματισμό εφαρμογής
    if st.button("❌ Κλείσιμο Εφαρμογής"):
        st.write("Η εφαρμογή τερματίζεται...")
        os.system("taskkill /F /IM python.exe")  # Windows
        os.system("pkill -f streamlit")  # Linux/macOS
        os._exit(0)