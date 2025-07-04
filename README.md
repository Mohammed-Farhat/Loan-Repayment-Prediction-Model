# Loan-Repayment-Prediction-Model

This project aims to predict whether a borrower is likely to repay or default on a loan using historical LendingClub data. We explore the full machine learning pipeline—from data cleaning and preprocessing to building and evaluating classification models, including a Neural Network and a Random Forest Classifier.

## 📁 Dataset

The dataset comes from LendingClub and contains detailed information on loans, including:

- Loan amount, interest rate, term
- Borrower information (employment length, home ownership, income)
- Credit history (open accounts, revolving utilization, public records)
- Loan status (Fully Paid or Charged Off)

Target Variable: `loan_status`  
Objective: Predict whether the loan will be repaid (`Fully Paid`) or defaulted (`Charged Off`).

---

## 🛠️ Project Pipeline

### 🔍 Step 1: Exploratory Data Analysis (EDA)
- Countplot of loan status
- Correlation heatmap of numeric features
- Loan amount distribution and boxplots by loan status
- Analysis of subgrades and employment length

### 🧹 Step 2: Data Cleaning & Preprocessing
- Handling missing values (emp_title, mort_acc, etc.)
- Dropping irrelevant features
- Feature engineering (e.g. extracting zip codes)
- Converting categorical features into dummy variables

### 🔄 Step 3: Train/Test Split
- Using `train_test_split` from `sklearn`
- 80/20 split for training and evaluation

### 🔧 Step 4: Feature Scaling
- Scaling all numeric features using `MinMaxScaler` to normalize inputs for the ANN

### 🧠 Step 5: Artificial Neural Network (ANN)
- Built with TensorFlow/Keras
- Architecture: Dense layers + Dropout for regularization
- Trained using binary crossentropy loss

### 🌲 Step 6: Random Forest Classifier
- Implemented using `sklearn`
- `GridSearchCV` used for hyperparameter tuning
- Handled class imbalance with `class_weight='balanced'`

### 🧪 Step 7: Support Vector Machine (SVM)
- Will train and evaluate an SVM classifier for comparison

### 🧪 Step 8: Evaluation
- Confusion matrix and classification report
- Comparison between ANN and RF predictions
- Analysis of performance (precision, recall, f1-score)

### ❓ Step 9: Predicting on a New Customer
- Select a random customer
- Predict whether they will repay the loan
- Compare prediction to actual outcome


---

## 📊 Performance Metrics

Metrics used to evaluate model performance:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Validation Loss Curve

---

## 📦 Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow / keras