import pandas as pd
import os
from modules.preprocessing import clean_and_preprocess_data, get_preprocessing_summary

def load_data(filepath='./dataset/accident.csv'):
    """
    Load the accident data from CSV file
    """
    try:
        # Load the raw data
        raw_df = pd.read_csv(filepath)
        
        # Record initial stats
        rows_before = len(raw_df)
        cols_before = len(raw_df.columns)
        missing_before = raw_df.isnull().sum().sum()
        
        # Process data (fill missing values, create features)
        processed_df = raw_df.copy()
        
        # Fill missing values
        processed_df['Gender'].fillna(processed_df['Gender'].mode()[0], inplace=True)
        processed_df['Speed_of_Impact'].fillna(processed_df['Speed_of_Impact'].median(), inplace=True)
        
        # Create additional features
        processed_df['Age_Group'] = pd.cut(processed_df['Age'], 
                                         bins=[0, 30, 45, 60, 100], 
                                         labels=['<30', '30-45', '45-60', '>60'])
        processed_df['Speed_Group'] = pd.cut(processed_df['Speed_of_Impact'], 
                                           bins=[0, 40, 80, 120], 
                                           labels=['Low', 'Medium', 'High'])
        processed_df['Age_Speed_Ratio'] = processed_df['Age'] / processed_df['Speed_of_Impact']
        processed_df['Safety_Score'] = ((processed_df['Helmet_Used'] == 'Yes').astype(int) + 
                                       (processed_df['Seatbelt_Used'] == 'Yes').astype(int))
        processed_df['Age_Speed_Ratio_Bin'] = pd.qcut(processed_df['Age_Speed_Ratio'], 
                                                    4, 
                                                    labels=['Very Low', 'Low', 'High', 'Very High'])
        
        # Record final stats
        rows_after = len(processed_df)
        cols_after = len(processed_df.columns)
        missing_after = processed_df.isnull().sum().sum()
        
        # Create summary
        preprocessing_summary = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'cols_before': cols_before,
            'cols_after': cols_after,
            'missing_before': missing_before,
            'missing_after': missing_after,
            'new_columns': list(set(processed_df.columns) - set(raw_df.columns))
        }
        
        # Add info about specific columns cleaned
        if raw_df['Gender'].isnull().any():
            preprocessing_summary['gender_filled'] = raw_df['Gender'].isnull().sum()
        
        if raw_df['Speed_of_Impact'].isnull().any():
            preprocessing_summary['speed_filled'] = raw_df['Speed_of_Impact'].isnull().sum()
        
        return processed_df, preprocessing_summary
        
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None