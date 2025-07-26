# train.pyghj edited 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
os.makedirs("app", exist_ok=True)
data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'app/model.joblib')
