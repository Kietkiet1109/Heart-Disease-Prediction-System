# Heart Disease Prediction System

## 1. Project Overview
Heart disease remains one of the leading causes of death globally, making early identification of high-risk individuals essential for timely prevention and treatment. This project develops a machine learning pipeline to predict whether an individual is at **High Risk** of heart-related issues using a rich dataset of health indicators.

The system benchmarks over 10 algorithms and implements a **Stacked Generalization (Stacking)** architecture to maximize prediction stability and minimize false negatives.

## 2. Dataset Statistics
The dataset contains demographic, behavioral, and medical history features from **315,607 individuals**.

* **Total Observations:** 315,607
* **Features:** 39 (e.g., BMI, Smoking, Alcohol, Stroke History, Physical Health Days)
* **Target:** `HighRisk` (Binary: Yes/No)
* **Class Imbalance:** Only ~4.36% of the original dataset was classified as "High Risk". **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the training data.

## 3. Methodology
The project follows a modular Data Science pipeline:
1.  **Data Preprocessing:** Cleaning missing values, converting categorical variables to dummy/indicator variables.
2.  **Feature Engineering:** Scaling numerical features and applying SMOTE to handle class imbalance.
3.  **Model Selection:** Evaluated models including Logistic Regression, Random Forest, XGBoost, and MLP.
4.  **Advanced Architecture:**
    * **ANN (Artificial Neural Network):** Tuned for high recall.
    * **Stacked Model:** Combines predictions from weak learners (Logistic, KNN, etc.) to improve final accuracy.

## 4. Model Performance
The **Stacked Model** and **ANN** demonstrated the best performance on the test set, effectively identifying high-risk patients.

| Model Type | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Stacked Model** | **0.9715** | 0.9741 | **0.9969** | **0.9854** |
| **ANN Model** | 0.9714 | 0.9802 | 0.9903 | 0.9852 |
| Random Forest | 0.9716 | 0.9737 | 0.9974 | 0.9854 |
| Logistic Regression | 0.9680 | 0.9724 | 0.9949 | 0.9835 |
| XGBoost | 0.9627 | 0.9628 | 0.9999 | 0.9810 |

*Note: The Stacked Model achieved the highest Recall (Sensitivity), which is crucial in medical diagnosis to avoid missing positive cases.*

## 5. Installation & Usage

### Prerequisites
* Python 3.10+
* TensorFlow / Keras
* Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/Kietkiet1109/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System

# 2. Install dependencies
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn xgboost
```

## 6. Execution Steps
Run the scripts in the following order to reproduce the results:

1. **Data Preprocessing:** Clean raw data and transform features for modeling.
```bash
python data_cleaning.py
python data_transformation.py
```

2. **Analysis & Visualization:** Generate statistical reports and heatmaps to understand data distribution.
```bash
python data_evaluate.py
```

3. **Train Models:** Train the Neural Network and the Stacking Ensemble classifier.
```bash
python ann_model.py
python stacked_model.py
python mlp_classifier.py
```

4. **Prediction:** Run the final inference script using the best performing models.
```bash
python heartrisk_predictions.py
```