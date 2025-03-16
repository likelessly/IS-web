import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as XGBoost

def prepare_model_data(df):
    """
    Prepare data for modeling
    """
    df_model = pd.get_dummies(df, columns=['Gender', 'Helmet_Used', 'Seatbelt_Used'], drop_first=True)
    
    # Define features and target
    X = df_model[['Age', 'Speed_of_Impact', 'Gender_Male', 'Helmet_Used_Yes', 
                 'Seatbelt_Used_Yes', 'Age_Speed_Ratio', 'Safety_Score']]
    y = df_model['Survived']
    
    return X, y

def train_models(X, y):
    """
    Train multiple models and compare performance
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Build models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBoost.XGBClassifier(random_state=42)
    }
    
    # Track results
    results = {}
    accuracies = []
    aucs = []
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'model': model
        }
        
        accuracies.append(accuracy)
        aucs.append(auc)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': list(models.keys()),
        'Accuracy': accuracies,
        'AUC Score': aucs
    })
    
    # Find best model based on AUC
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best_model = results[best_model_name]['model']
    
    return results, comparison_df, best_model_name, best_model

def plot_model_comparison(comparison_df):
    """
    Create bar chart comparing model performances
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Accuracy'], width, label='Accuracy')
    ax.bar(x + width/2, comparison_df['AUC Score'], width, label='AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend()
    plt.xticks(rotation=45)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    
    return fig

def plot_feature_importance(model, model_name, feature_names):
    """
    Plot feature importance or coefficients
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if model_name == 'Logistic Regression':
        # For logistic regression, show coefficients
        coefs = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', ascending=False)
        
        sns.barplot(x='Coefficient', y='Feature', data=coefs, ax=ax)
        plt.title('Feature Coefficients')
        plt.axvline(x=0, color='r', linestyle='--')
        
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models, show feature importance
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=importances, ax=ax)
        plt.title(f'Feature Importance ({model_name})')
    
    return fig

def train_default_models(df):
    """
    Train default models on the full dataset
    """
    X, y = prepare_model_data(df)
    
    # Build models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBoost.XGBClassifier(random_state=42)
    }
    
    # Train all models on the full dataset
    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
    
    return trained_models, X.columns