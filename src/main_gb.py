# %%
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_experiment("model-gb")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# %%
data = pd.read_csv("data/water_potability.csv")

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(train_data.shape, test_data.shape)

# %%
def fill_missing_with_median(df):
    for col in df.columns:
        df.fillna({col: df[col].median()}, inplace=True)
    return df

train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# %%
from sklearn.ensemble import GradientBoostingClassifier
import pickle
X_train = train_processed_data.drop("Potability", axis=1)
y_train = train_processed_data["Potability"]

n_estimators = 1000
with mlflow.start_run():
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)

    pickle.dump(clf, open("model.pkl", "wb"))

    # %%
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    model = pickle.load(open("model.pkl", "rb"))
    X_test = test_processed_data.drop("Potability", axis=1)
    y_test = test_processed_data["Potability"]

    y_pred = model.predict(X_test) 

    mlflow.log_metric("accuracy_score", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_param("n_estimators", n_estimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(model, "model")

    mlflow.log_artifact(__file__)

    mlflow.set_tags({"model": "GradientBoostingClassifier", "author": "tqiu"})

