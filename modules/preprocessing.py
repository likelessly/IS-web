import pandas as pd
import numpy as np

def clean_and_preprocess_data(df):
    """
    Clean and preprocess the accident dataset
    
    Parameters:
    df (DataFrame): The raw dataframe
    
    Returns:
    DataFrame: The cleaned and preprocessed dataframe
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # 1. Handle Missing Values
    print(f"Missing values before cleaning: {processed_df.isnull().sum().sum()}")
    
    # Fill missing Gender values with mode
    if 'Gender' in processed_df.columns and processed_df['Gender'].isnull().any():
        mode_gender = processed_df['Gender'].mode()[0]
        processed_df['Gender'].fillna(mode_gender, inplace=True)
        
    # Fill missing Speed_of_Impact values with median
    if 'Speed_of_Impact' in processed_df.columns and processed_df['Speed_of_Impact'].isnull().any():
        median_speed = processed_df['Speed_of_Impact'].median()
        processed_df['Speed_of_Impact'].fillna(median_speed, inplace=True)
    
    print(f"Missing values after cleaning: {processed_df.isnull().sum().sum()}")
    
    # 2. Feature Engineering
    # Create age groups
    processed_df['Age_Group'] = pd.cut(
        processed_df['Age'], 
        bins=[0, 30, 45, 60, 100], 
        labels=['<30', '30-45', '45-60', '>60']
    )
    
    # Create speed groups
    processed_df['Speed_Group'] = pd.cut(
        processed_df['Speed_of_Impact'], 
        bins=[0, 40, 80, 120], 
        labels=['Low', 'Medium', 'High']
    )
    
    # Calculate age/speed ratio (protective measure)
    processed_df['Age_Speed_Ratio'] = processed_df['Age'] / processed_df['Speed_of_Impact']
    
    # Create safety score (count of safety measures)
    processed_df['Safety_Score'] = (
        (processed_df['Helmet_Used'] == 'Yes').astype(int) + 
        (processed_df['Seatbelt_Used'] == 'Yes').astype(int)
    )
    
    # Bin the age/speed ratio into quartiles
    processed_df['Age_Speed_Ratio_Bin'] = pd.qcut(
        processed_df['Age_Speed_Ratio'], 
        4, 
        labels=['Very Low', 'Low', 'High', 'Very High']
    )
    
    return processed_df

def get_preprocessing_summary(df_before, df_after):
    """
    Generate a summary of preprocessing steps and changes
    
    Parameters:
    df_before (DataFrame): The raw dataframe
    df_after (DataFrame): The processed dataframe
    
    Returns:
    dict: Summary of preprocessing steps and changes
    """
    summary = {
        'rows_before': len(df_before),
        'rows_after': len(df_after),
        'columns_before': len(df_before.columns),
        'columns_after': len(df_after.columns),
        'missing_values_before': df_before.isnull().sum().sum(),
        'missing_values_after': df_after.isnull().sum().sum(),
        'new_columns': list(set(df_after.columns) - set(df_before.columns)),
    }
    
    # If Gender was cleaned, add that info
    if 'Gender' in df_before.columns and df_before['Gender'].isnull().any():
        summary['gender_filled'] = df_before['Gender'].isnull().sum()
        
    # If Speed was cleaned, add that info
    if 'Speed_of_Impact' in df_before.columns and df_before['Speed_of_Impact'].isnull().any():
        summary['speed_filled'] = df_before['Speed_of_Impact'].isnull().sum()
    
    return summary