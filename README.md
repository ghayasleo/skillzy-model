# 🎯 Career Role Prediction Model

This repository contains a machine learning-based career recommendation system. It uses a **Random Forest Classifier** trained on structured survey data (27 questions) to predict suitable career roles for users based on their skills, personality traits, and preferences.

---

## 🧠 Model Summary

- **Algorithm:** Random Forest Classifier
- **Input:** 27-question answers (1–7 scale)
  - 17 Technical & Soft Skills (used as-is)
  - 10 Personality & Values (scaled to 0–1)
- **Output:** One of 16 predefined career roles
- **Accuracy:** ~100% on test set

---

## 🚀 How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py
```

### 3. Save Model Files

```bash
python src/save_model.py
```

### 4. Make Predictions

Edit `predict_role.py` to include your 27-item input array:

```bash
# Example input (1–7 scale)
input_array = [5, 6, 4, ..., 7]
```

Then run:

```bash
python src/predict_role.py
```

### 5. Evaluate Model

To generate the confusion matrix:

```bash
python src/confusion-matrix.py
```

To visualize feature importance:

```bash
python src/feature_importance.py
```

---

## 📊 Input Format

Each input must be a list of 27 values from 1 to 7:

```bash
[
  7, 4, 5, 6, 3, 2, 5, ..., 6  # total 27 items
]
```

- First 17 values are used directly
- Last 10 values are scaled: (value - 1) / 6

---

## ✅ Example Prediction

```bash
predict_role_from_frontend([
  5, 7, 2, 6, 7, 7, 6, 6, 4, 6, 6, 2, 5, 2, 7, 1, 6,
  3, 4, 5, 7, 6, 7, 7, 7, 4, 1
])
# → 'Graphics Designer'
```

---

## 🧪 Testing & Validation

To verify if the model is making correct predictions on real-style inputs (like those coming from your frontend), use:

```bash
python src/validate.py <index>
```

You can pass index between 1-9179
