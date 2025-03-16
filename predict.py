import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

def load_model(model_name, models_dir='./models'):
    """
    Load a saved model from disk
    
    Parameters:
    model_name (str): Name of the model to load (without .joblib extension)
    models_dir (str): Directory where models are saved
    
    Returns:
    Model object
    """
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    return joblib.load(model_path)

def load_all_models(models_dir='./models'):
    """
    Load all saved models from the models directory
    
    Parameters:
    models_dir (str): Directory where models are saved
    
    Returns:
    Dictionary of model objects
    """
    models = {}
    
    # Get all .joblib files that aren't feature_names.joblib
    for filename in os.listdir(models_dir):
        if filename.endswith('.joblib') and filename != 'feature_names.joblib':
            model_name = filename.replace('.joblib', '')
            models[model_name] = joblib.load(os.path.join(models_dir, filename))
    
    return models

def load_feature_names(models_dir='./models'):
    """
    Load feature names for the model
    
    Parameters:
    models_dir (str): Directory where feature names are saved
    
    Returns:
    List of feature names
    """
    feature_path = os.path.join(models_dir, "feature_names.joblib")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature names not found at {feature_path}")
    
    return joblib.load(feature_path)

def prepare_input_features(age, gender, speed, helmet_used, seatbelt_used):
    """
    Prepare input features for prediction
    
    Parameters:
    age (int): Age of the person
    gender (str): 'Male' or 'Female'
    speed (float): Speed of impact
    helmet_used (str): 'Yes' or 'No'
    seatbelt_used (str): 'Yes' or 'No'
    
    Returns:
    numpy.ndarray: Features array for prediction
    """
    # Convert categorical values to binary
    gender_male = 1 if gender == "Male" else 0
    helmet_yes = 1 if helmet_used == "Yes" else 0
    seatbelt_yes = 1 if seatbelt_used == "Yes" else 0
    
    # Compute derived features
    age_speed_ratio = age / speed if speed > 0 else 0
    safety_score = helmet_yes + seatbelt_yes
    
    # Create feature array
    features = np.array([[age, speed, gender_male, helmet_yes, seatbelt_yes, 
                         age_speed_ratio, safety_score]])
    
    return features

def predict_survival(model, features):
    """
    Make survival prediction using the trained model
    """
    try:
        # Convert features to numpy array if needed
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        else:
            features_array = np.array(features).reshape(1, -1)
            
        # Check if model is a dictionary (happens sometimes with serialization)
        if isinstance(model, dict) and hasattr(model.get('model', None), 'predict_proba'):
            # Extract the actual model from the dictionary
            actual_model = model['model']
            survival_prob = actual_model.predict_proba(features_array)[0, 1]
        # Check if model has predict_proba method (sklearn models)
        elif hasattr(model, 'predict_proba'):
            survival_prob = model.predict_proba(features_array)[0, 1]
        # For models with only predict method
        else:
            prediction = model.predict(features_array)
            # If prediction is a probability
            if prediction.ndim > 1 and prediction.shape[1] > 1:
                survival_prob = prediction[0, 1]
            else:
                survival_prob = prediction[0]  # Binary prediction (0 or 1)
        
        # Determine survival class
        survival_class = "Likely to Survive" if survival_prob >= 0.5 else "Unlikely to Survive"
        
        return survival_prob, survival_class
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        # Return default values
        return 0.5, "Prediction error"

def create_probability_gauge(survival_prob):
    """
    Create a gauge chart showing the survival probability
    
    Parameters:
    survival_prob (float): Probability between 0 and 1
    
    Returns:
    matplotlib.figure.Figure: Figure object with the gauge chart
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Set up gauge chart
    gauge_colors = ["#FF0000", "#FFA500", "#FFFF00", "#008000"]
    n_colors = len(gauge_colors)
    bounds = np.linspace(0, 1, n_colors + 1)
    
    # Draw gauge background
    for i in range(n_colors):
        ax.axvspan(bounds[i], bounds[i+1], facecolor=gauge_colors[i], alpha=0.3)
    
    # Add arrow to show probability
    ax.arrow(0, 0, survival_prob, 0, head_width=0.1, head_length=0.05, 
             fc='black', ec='black', length_includes_head=True)
    
    # Add text
    ax.text(survival_prob, 0.2, f"{survival_prob:.1%}", 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Customize plot
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.title("Survival Probability")
    
    return fig

def generate_safety_recommendations(age, speed, safety_score, avg_survival_rate=0.5):
    """
    Generate safety recommendations based on input parameters
    
    Parameters:
    age (int): Age of the person
    speed (float): Speed of impact
    safety_score (int): Number of safety features used (0-2)
    avg_survival_rate (float): Average survival rate from the dataset
    
    Returns:
    list: Safety recommendations
    """
    recommendations = []
    
    if safety_score < 2:
        recommendations.append("⚠️ Always use both helmet and seatbelt for maximum protection.")
    
    if speed > 60:
        recommendations.append(f"⚠️ Your speed ({speed} km/h) is high. Lower speeds significantly increase survival chances.")
    
    if age > 60:
        recommendations.append("⚠️ For older individuals, extra caution and safety measures are recommended.")
    
    recommendations.append(f"For reference, the average survival rate in our dataset is {avg_survival_rate:.1%}.")
    
    return recommendations

def main():
    """
    Example usage of prediction functions
    """
    try:
        # Load models
        print("Loading models...")
        models = load_all_models()
        
        if not models:
            print("No models found. Please run train_models.py first.")
            return
        
        # Example prediction
        print("\nMaking a sample prediction:")
        age = 35
        gender = "Male"
        speed = 60
        helmet_used = "Yes"
        seatbelt_used = "Yes"
        
        print(f"\nInput parameters:")
        print(f"Age: {age}")
        print(f"Gender: {gender}")
        print(f"Speed of Impact: {speed} km/h")
        print(f"Helmet Used: {helmet_used}")
        print(f"Seatbelt Used: {seatbelt_used}")
        
        # Prepare features
        features = prepare_input_features(age, gender, speed, helmet_used, seatbelt_used)
        
        # Make predictions with each model
        print("\nPredictions:")
        print("-" * 40)
        
        for name, model in models.items():
            prob, result = predict_survival(model, features)
            print(f"{name}: {prob:.2%} ({result})")
        
        # Safety recommendations
        print("\nSafety Recommendations:")
        print("-" * 40)
        
        safety_score = (helmet_used == "Yes") + (seatbelt_used == "Yes")
        recommendations = generate_safety_recommendations(age, speed, safety_score)
        
        for rec in recommendations:
            print(rec)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()