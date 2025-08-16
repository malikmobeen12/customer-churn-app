# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, roc_curve)
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

# Load data
def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

# Feature engineering
def feature_engineering(df):
    # Create service aggregations
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = df[service_cols].apply(
        lambda x: sum([1 if val == 'Yes' else 0 for val in x]), axis=1)
    
    # Create tenure groups
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                              labels=['0-1Y', '1-2Y', '2-4Y', '4-6Y'])
    
    # Create interaction features
    df['MonthlyTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['TotalMonthlyRatio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
    
    return df

# Preprocessing pipeline
def create_preprocessor():
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                       'MonthlyTenureRatio', 'TotalMonthlyRatio', 'ServiceCount']
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                           'PhoneService', 'MultipleLines', 'InternetService',
                           'Contract', 'PaperlessBilling', 'PaymentMethod',
                           'TenureGroup']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

# Calculate class weights for imbalance handling
def get_class_weights(y):
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

# Model training and evaluation
def main():
    # Load and prepare data
    df = load_data()
    df = feature_engineering(df)
    
    # Split data
    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing
    preprocessor = create_preprocessor()
    
    # Calculate class weights
    class_weights = get_class_weights(y_train)
    xgb_scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # Define models with hyperparameter grids
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['lbfgs', 'liblinear']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(class_weight='balanced'),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_split': [5, 10]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                  scale_pos_weight=xgb_scale_pos_weight),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__subsample': [0.8, 1.0]
            }
        }
    }
    
    # Train and evaluate with hyperparameter tuning
    results = {}
    best_score = 0
    best_model = None
    
    for name, config in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        # Grid search
        grid = GridSearchCV(pipeline, config['params'], cv=StratifiedKFold(n_splits=5),
                           scoring='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        # Best model from grid search
        best_pipeline = grid.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        }
        
        results[name] = metrics
        print(f"{name} - Best Params: {grid.best_params_}")
        print(f"{name} - Metrics: {metrics}")
        
        # Save best model
        if metrics['ROC AUC'] > best_score:
            best_score = metrics['ROC AUC']
            best_model = best_pipeline
    
    # Print results
    print(pd.DataFrame(results).T)
    
    # Save best model
    joblib.dump(best_model, 'churn_model.pkl')
    print(f"Best model saved (ROC AUC: {best_score:.4f})")
    
    # Generate ROC curve
    plt.figure(figsize=(10, 8))
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    main()