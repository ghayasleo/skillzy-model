from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from train import X_test, y_test

model = joblib.load('./src/assets/random_forest_model.pkl')
le = joblib.load('./src/assets/label_encoder.pkl')

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation='vertical')
plt.tight_layout()
plt.show()