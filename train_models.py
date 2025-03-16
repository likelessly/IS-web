import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as XGBoost
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='./dataset/accident.csv'):
    """
    Load and preprocess the accident data
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    
    # Handle missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Speed_of_Impact'].fillna(df['Speed_of_Impact'].median(), inplace=True)
    
    # Create additional features
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '>60'])
    df['Speed_Group'] = pd.cut(df['Speed_of_Impact'], bins=[0, 40, 80, 120], labels=['Low', 'Medium', 'High'])
    df['Age_Speed_Ratio'] = df['Age'] / df['Speed_of_Impact']
    df['Safety_Score'] = (df['Helmet_Used'] == 'Yes').astype(int) + (df['Seatbelt_Used'] == 'Yes').astype(int)
    df['Age_Speed_Ratio_Bin'] = pd.qcut(df['Age_Speed_Ratio'], 4, labels=['Very Low', 'Low', 'High', 'Very High'])
    
    return df

def prepare_features(df):
    """
    Prepare features for modeling
    """
    # One-hot encode categorical variables
    df_model = pd.get_dummies(df, columns=['Gender', 'Helmet_Used', 'Seatbelt_Used'], drop_first=True)
    
    # Define features and target
    X = df_model[['Age', 'Speed_of_Impact', 'Gender_Male', 'Helmet_Used_Yes', 
                 'Seatbelt_Used_Yes', 'Age_Speed_Ratio', 'Safety_Score']]
    y = df_model['Survived']
    
    return X, y

def train_and_evaluate_models(X, y):
    """
    Train multiple models and evaluate performance
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define models to train
    models = {
        'Logistic_Regression': LogisticRegression(random_state=42),
        'Random_Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBoost.XGBClassifier(random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    
    print("Training and evaluating models:")
    print("-" * 40)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc
        }
        
        print(f"{name}: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")
    
    # Find best model based on AUC
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best_model = results[best_model_name]['model']
    best_auc = results[best_model_name]['auc']
    
    print(f"\nBest model: {best_model_name} with AUC = {best_auc:.4f}")
    
    return results, best_model_name, best_model

def save_models(models_dict, output_dir='./models'):
    """
    Save trained models to disk
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save each model
    for name, model_data in models_dict.items():
        model = model_data['model']
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Model {name} saved to {model_path}")
    
    # Save model metrics
    metrics = {name: {'accuracy': data['accuracy'], 'auc': data['auc']} 
               for name, data in models_dict.items()}
    
    metrics_df = pd.DataFrame({
        'Model': list(metrics.keys()),
        'Accuracy': [metrics[m]['accuracy'] for m in metrics],
        'AUC': [metrics[m]['auc'] for m in metrics]
    })
    
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Model metrics saved to {metrics_path}")

def save_feature_names(X, output_dir='./models'):
    """
    Save feature names for later use in prediction
    """
    feature_path = os.path.join(output_dir, "feature_names.joblib")
    joblib.dump(list(X.columns), feature_path)
    print(f"Feature names saved to {feature_path}")

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    if df is None:
        return
    
    # Prepare features
    print("Preparing features...")
    X, y = prepare_features(df)
    
    # Train and evaluate models
    print("Training models...")
    results, best_model_name, best_model = train_and_evaluate_models(X, y)
    
    # Save models
    print("\nSaving models...")
    save_models(results)
    
    # Save feature names
    save_feature_names(X)
    
    print("\nModel training and saving complete!")
    print(f"Best model: {best_model_name}")

if __name__ == "__main__":
    main()