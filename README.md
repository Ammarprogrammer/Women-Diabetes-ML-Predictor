# 🩺 Women Diabetes Prediction using Decision Tree Classifier

This project predicts whether a woman is diabetic or not based on medical attributes using **DecisionTreeClassifier**.  
The model is trained with **balanced data (SMOTE)** and evaluated using **ROC-AUC Score**, **Confusion Matrix**, and **Classification Report**.

---

## 📊 Features Used
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

Target column:
- `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

---

## ⚙️ Workflow
1. **Ignore Warnings**  
   - Suppress all warnings for clean output.  

2. **Import Libraries**
   - pandas, numpy, matplotlib  
   - sklearn → StandardScaler, train_test_split, DecisionTreeClassifier, cross_val_score, classification_report, confusion_matrix, roc_auc_score  
   - imblearn → SMOTE  

3. **Data Preprocessing**
   - Load dataset into a DataFrame  
   - Standardize data with **StandardScaler** (scaled between -1 and 1)  

4. **Train-Test Split**
   - 80% for training, 20% for testing  

5. **Handle Class Imbalance**
   - Use **SMOTE** to oversample the minority class  

6. **Model Training**
   - Algorithm: **DecisionTreeClassifier**  
     ```python
     DecisionTreeClassifier(
       max_depth=5,
       min_samples_split=10,
       min_samples_leaf=5,
       class_weight="balanced",
       random_state=42
     )
     ```

7. **Model Evaluation**
   - **Cross Validation Score** (Mean & Std Accuracy)  
   - **Classification Report**  
   - **Confusion Matrix**  
   - **ROC-AUC Score**

8. **Prediction**
   - Predict outcomes using `X_test`  
   - Accept user input for new data and classify as **Diabetic / Non-Diabetic**

---
<img width="1366" height="768" alt="2025-10-06" src="https://github.com/user-attachments/assets/6bd79835-ca16-4761-bad4-53cf770531e5" />
