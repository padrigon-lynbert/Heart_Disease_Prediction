
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, precision_recall_curve

from icecream import ic as print

# locating filepath of the dataframe
base_path = "D:\manualCDmanagement\codes\Projects\VMs\skl algorithms\Logistic Regression\Heart_disease.1/Storage"
file_name = "heart_disease_dataset.csv"
file_path = os.path.join(base_path, file_name)

df = pd.read_csv(file_path) 

# data cleaning

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Smoking'] = df['Smoking'].map({'Current': 2, 'Never': 0, "Former": 1})
df['Family History'] = df['Family History'].map({'Yes': 1, 'No': 0})
df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})
df['Obesity'] = df['Obesity'].map({'Yes': 1, 'No': 0})
df['Exercise Induced Angina'] = df['Exercise Induced Angina'].map({'Yes': 1, 'No': 0})
df['Chest Pain Type'] = df['Chest Pain Type'].map({'Typical Angina': 3, 'Atypical Angina': 2, 'Non-anginal Pain': 1, 'Asymptomatic': 0})
df['Alcohol Intake'] = df['Alcohol Intake'].fillna('None').map({'Heavy': 2, 'Moderate': 1, 'None':1})

# partitioning, modeling

X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Predict probabilities instead of classes
y_probs = model.predict_proba(X_test)[:, 1]
threshold = 0.7
y_pred_adjusted = (y_probs >= threshold).astype(int)

# Evaluate adjusted predictions
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
class_report_adjusted = classification_report(y_test, y_pred_adjusted)

print("Confusion Matrix (Adjusted):")
print(conf_matrix_adjusted)
print("\nClassification Report (Adjusted):")
print(class_report_adjusted)



precision, recall, threshold = precision_recall_curve(y_test, y_probs)
area_under_pr = auc(recall, precision)

# # Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='b', label=f'Precision-Recall Curve (AUC = {area_under_pr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)



# # Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_adjusted, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix 0.7 Threshold')

# # plt.show()




# Create a base heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_adjusted, annot=True, fmt='d', cbar=False, square=True,
            cmap='Blues', linewidths=0.5, annot_kws={"size": 16})

# Overlay colored rectangles for each type of cell
colors = {
    'TP': 'blue',
    'TN': 'blue',
    'FP': 'red',
    'FN': 'orange'
}

# Define the coordinates for each cell type
cell_types = {
    'TN': (1, 1),
    'FP': (0, 1),
    'FN': (1, 0),
    'TP': (0, 0)
}

# Add colored rectangles
for cell_type, (row, col) in cell_types.items():
    plt.gca().add_patch(plt.Rectangle((col, row), 1, 1, fill=True, color=colors[cell_type], alpha=0.3))

plt.title('Confusion Matrix Threshold 0.7')



# plot coefficients

coefficients = model.coef_[0]

feature_names = X.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

# Plot coefficients
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
# plt.xlabel('Coefficient Magnitude')
# plt.ylabel('Feature')
# plt.title('Heart Disease Coefficients')
# plt.grid(True)

# Save image
fig_name = "Confusion Matrix Threshold 0.7.png"
bp = base_path + "/Figures"
save_to = os.path.join(bp, fig_name)
# plt.savefig(save_to)
# plt.savefig(save_to, dpi=300, bbox_inches='tight')
plt.savefig(save_to, bbox_inches='tight')

# plt.show()