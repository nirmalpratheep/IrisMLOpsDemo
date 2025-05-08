
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow

def train():
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    X = df.drop("species", axis=1)
    y = df["species"]

    clf = RandomForestClassifier()
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)

    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_metric("accuracy", acc)
    joblib.dump(clf, "iris_model.pkl")
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    train()
