import joblib
import numpy as np
import pandas as pd

df = pd.read_csv('./src/assets/career-mapping.csv')
X = df.drop(columns=['Role'])

model = joblib.load('./src/assets/random_forest_model.pkl')
le = joblib.load('./src/assets/label_encoder.pkl')

def predict_role(input_array):
    if len(input_array) != 27:
        return "Error: Input must contain exactly 27 elements."

    # First 17 are technical/soft skill ratings (1-7 scale) – use directly
    tech_soft_skills = input_array[:17]

    # Last 10 are psychological/values – scale 1-7 to 0–1
    psych_values = [(val - 1) / 6 for val in input_array[17:]]  # Normalize 1-7 to 0.0–1.0

    full_input = np.array(tech_soft_skills + psych_values).reshape(1, -1)
    # print("Input for prediction:", full_input)
    prediction = model.predict(pd.DataFrame(full_input, columns=X.columns))
    predicted_role = le.inverse_transform(prediction)[0]
    print(f"Predicted Role: {predicted_role}")

    return predicted_role

# predict_role([7, 1, 2, 2, 3, 1, 4, 5, 2, 3, 2, 5, 1, 7, 2, 3, 4, 3, 3, 4, 5, 5, 3, 1, 7, 5, 7])
# predict_role([5, 7, 2, 6, 7, 7, 6, 6, 4, 6, 6, 2, 5, 2, 7, 1, 6, 3, 4, 5, 7, 6, 7, 7, 7, 4, 1])
# predict_role([7, 3, 6, 3, 1, 5, 6, 3, 1, 4, 3, 7, 3, 6, 5, 7, 2, 7, 1, 7, 5, 7, 2, 3, 3, 3, 1])