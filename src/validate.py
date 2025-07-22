import pandas as pd
from predict_role import predict_role

# Split features and target
df = pd.read_csv('./src/career-mapping.csv')

# Choose a few rows from original df
sample = df.iloc[4560]  # index
input_array = sample.drop('Role').tolist()

# Manually scale last 10 elements like frontend
frontend_style_input = input_array[:17] + [round(v * 6 + 1) for v in input_array[17:]]  # Rescale 0-1 back to 1-7

res = predict_role(frontend_style_input) == sample["Role"]
print(f"Validation result: {res}")