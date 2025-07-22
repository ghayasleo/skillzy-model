import sys
import pandas as pd
from predict_role import predict_role

print("🔍 Validating model predictions...")

# Check if index was provided
if len(sys.argv) < 2:
    print("❌ Usage: python src/validate.py <index>")
    sys.exit(1)

try:
    index = int(sys.argv[1])
except ValueError:
    print("❌ Error: Index must be an integer.")
    sys.exit(1)

# Load dataset
df = pd.read_csv('./src/assets/career-mapping.csv')

# Validate index range
if index < 0 or index >= len(df):
    print(f"❌ Error: Index out of range. Must be between 0 and {len(df) - 1}")
    sys.exit(1)

# Get sample and simulate frontend input
sample = df.iloc[index]
input_array = sample.drop('Role').tolist()
frontend_style_input = input_array[:17] + [round(v * 6 + 1) for v in input_array[17:]]

# Predict and compare
predicted = predict_role(frontend_style_input)
actual = sample["Role"]
res = predicted == actual

# Output results
print(f"\n🎯 Validation Result for Index {index}")
print(f"Predicted Role : {predicted}")
print(f"Actual Role    : {actual}")
print(f"✅ Match        : {res}")
