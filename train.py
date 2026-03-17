import joblib
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
 

def folders():

    folder = Path("models")
    if not folder.exists():
        os.makedirs(folder, exist_ok=True)
    model_file = Path(folder/"churn_model.pkl")
    if not model_file.exists():
        with open(model_file, "w") as f:
            pass
        print("file is created")

def train(): 
    parent_dir = Path(__file__).parent         
    df = pd.read_csv("data/churn_data.csv")
    x = df.drop(columns = ["churn","custormer_id"])
    y = df["churn"]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test,y_proba)

    print(f"accuracy:{accuracy:.4f}")
    print(f"roc{auc:.4f}")

    joblib.dump(model,'models/churn_model.pkl')
    print("successfully saved the model")

folders()
train()

