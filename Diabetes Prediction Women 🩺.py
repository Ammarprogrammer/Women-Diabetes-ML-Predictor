import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report , confusion_matrix,  roc_auc_score
from imblearn.over_sampling import SMOTE # imblearn = imbalanced-learn, a separate Python library
# built for handling imbalanced datasets.
# from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_csv('Machine Learning/diabetes.csv')


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
Scaler = StandardScaler()
df_scaled = data.copy()
df_scaled[features] = Scaler.fit_transform(data[features])

X = df_scaled[features]
Y = df_scaled['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2 , random_state=42)

model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, 
    class_weight="balanced", random_state=42)
# option 1
# Oversampling minority (SMOTE)
# SMOTE = Synthetic Minority Over-sampling Technique It creates new synthetic samples of the minority class (diabetic).
smote = SMOTE(random_state=42)
X_res, Y_res = smote.fit_resample(X_train, Y_train)

# Option 2
# Undersampling majority
# Removes some non-diabetic cases randomly & This makes the dataset smaller but balanced.
#rus = RandomUnderSampler(random_state=42)
#X_res, Y_res = rus.fit_resample(X_train, Y_train)


# Cross-validation (on original data, just to see baseline)
scores = cross_val_score(model, X, Y, cv=2)
print("Scores:", np.round(scores,2))
print("Standard Deviation:", np.std(scores))
print("Mean Accuracy:", np.round(scores.mean(),2))

# Train on resampled dataset
model.fit(X_res, Y_res)

# Evaluate on test set
y_pred = model.predict(X_test)
# 0.5 = random guessing (no skill), 0.7–0.8 = fair model (better than average), 0.8–0.9 = good model, >0.9 = excellent (rare in medical datasets).
y_pred_proba = model.predict_proba(X_test)[:,1]
print("ROC-AUC Score:", roc_auc_score(Y_test, y_pred_proba))
print('Classification report')
print(classification_report(Y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

print('----Check you are Diabetic patient or not----')
try:
    Preg = int(input('Enter your Pregnancies: '))
    Glucose= int(input('Enter Glucose Level: '))
    BloodPressure = int(input('Enter BloodPressure: '))
    SkinThickness = int(input('Enter Thickness of skin: '))
    Insulin= int(input('Enter Insulin: '))
    BoMI = float(input('Enter Body Mass Index: '))
    DiabetesPedigreeFunction= float(input('Enter DiabetesPedigreeFunction: '))
    Age = int(input('Enter your Age: '))
    

    user_input_df = pd.DataFrame([{
        'Pregnancies': Preg,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BoMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }])

    user_input_scaled = Scaler.transform(user_input_df)
    prediction = model.predict(user_input_scaled)[0]
    
    if prediction == 0:
        print('Prediction based on Patient data: You are Non-diabetic Patien')
    else:  
        print('Prediction based on Patient data: You are diabetic Patien')
except Exception as e:
    print(e)