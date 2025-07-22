import joblib
from train import le, model

# Save the model
joblib.dump(model, './src/assets/random_forest_model.pkl')

# Save the label encoder
joblib.dump(le, './src/assets/label_encoder.pkl')