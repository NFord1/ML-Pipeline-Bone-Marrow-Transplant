# Bone Marrow Transplant Machine Learning Pipeline

## Project Overview
This project implements a **machine learning pipeline** to analyse data from bone marrow transplants. The pipeline preprocesses the data, applies dimensionality reduction using **Principal Component Analysis (PCA)**, and trains a **Logistic Regression model** to predict patient survival outcomes. The script also includes hyperparameter optimisation using **GridSearchCV** to identify the best model configuration.

---

## Features
1. **Data Preprocessing**:
   - Conversion of all columns to numeric values.
   - Handling missing values:
     - Categorical data: Filled using the most frequent value.
     - Numerical data: Filled using the mean.
   - One-hot encoding for categorical variables with a drop-first strategy to avoid multicollinearity.
   - Standard scaling for numerical variables to normalise the data.

2. **Pipeline**:
   - Combines preprocessing steps with PCA and Logistic Regression into a single **`scikit-learn` Pipeline**.
   - Automates preprocessing, dimensionality reduction, and classification.

3. **Hyperparameter Tuning**:
   - Uses **GridSearchCV** to find the best hyperparameters for:
     - Regularisation strength (`C`) in Logistic Regression.
     - Number of PCA components (`n_components`).

4. **Evaluation**:
   - Reports accuracy on the test dataset.
   - Outputs the best model configuration and hyperparameters.

---

## Dataset
The script processes the dataset stored in an ARFF file (`bone-marrow.arff`). Key features include demographic information, medical conditions, and transplant characteristics. The target variable is `survival_status`.

### Example Data Insights:
- **Number of unique values per column**: Provides an overview of the dataset's categorical and numerical features.
- **Columns with missing values**: Identifies attributes that require imputation.

---

## Results
- **Pipeline Accuracy on Test Set**: ~78.95%
- **Best Model Accuracy on Test Set**: ~81.58%
- **Best Model Configuration**:
  - Classifier: Logistic Regression with `C = 1.0`.
  - PCA Components: 37.

---

## Prerequisites
Ensure you have the following libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`

Install them using:
```bash
pip install numpy pandas scikit-learn scipy
```
## Usage
1. Clone the repository:
2. Ensure the `bone-marrow.arff` file is in the same directory as `solution.py`.
3. Run the script:
   ```bash
   python solution.py
   ```
4. Observe the results in the terminal, including model accuracy and the best configuration.

## Warnings
The script may generate warnings during runtime, such as:
- **Unknown categories during one-hot encoding**: Indicates that unseen categories in the test set were encoded as all zeros. This does not affect the accuracy of the results.

