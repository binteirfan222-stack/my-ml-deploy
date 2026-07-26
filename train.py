import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import skops.io as sio

df = pd.read_csv("./Data/dataset.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=125)

model = RandomForestClassifier(n_estimators=100, random_state=125)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
with open("./Results/metrics.txt", "w") as f:
    f.write(f"Accuracy = {accuracy_score(y_test, predictions):.3f}\n")

sio.dump(model, "./Model/model_pipeline.skops")