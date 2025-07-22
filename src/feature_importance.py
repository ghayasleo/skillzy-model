import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

model = joblib.load('./src/assets/random_forest_model.pkl')

df = pd.read_csv('./src/assets/career-mapping.csv')
X = df.drop(columns=['Role'])

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.show()