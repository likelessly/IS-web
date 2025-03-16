import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.visualization import (
    create_correlation_heatmap, 
    plot_survival_correlations,
    create_pairplot, 
    plot_categorical_survival_analysis,
    plot_distribution,
    plot_age_speed_heatmap,
    plot_grouped_survival
)

def get_basic_stats(df):
    """Get basic statistics about the dataset"""
    stats = {
        'num_records': len(df),
        'features': ', '.join([col for col in df.columns if col != 'Survived']),
    }
    
    if 'Survived' in df.columns:
        stats['survived_count'] = df['Survived'].sum()
        stats['not_survived_count'] = len(df) - df['Survived'].sum()
        stats['survival_rate'] = 100 * df['Survived'].mean()
    
    return stats

def get_categorical_columns(df):
    """Get list of categorical columns"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_numerical_columns(df):
    """Get list of numerical columns"""
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def preprocess_data(df):
    """
    Create derived columns needed for analysis
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
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
    
    # Create age/speed ratio
    processed_df['Age_Speed_Ratio'] = processed_df['Age'] / processed_df['Speed_of_Impact']
    
    # Create safety score
    processed_df['Safety_Score'] = (
        (processed_df['Helmet_Used'] == 'Yes').astype(int) + 
        (processed_df['Seatbelt_Used'] == 'Yes').astype(int)
    )
    
    # Bin the age/speed ratio
    processed_df['Age_Speed_Ratio_Bin'] = pd.qcut(
        processed_df['Age_Speed_Ratio'], 
        4, 
        labels=['Very Low', 'Low', 'High', 'Very High']
    )
    
    return processed_df

def analyze_safety_effect(df, category_col, target_col, title=None):
    """
    Analyze the effect of a categorical variable on the target
    
    Parameters:
    df (DataFrame): The dataframe
    category_col (str): Name of the categorical column
    target_col (str): Name of the target column
    title (str): Plot title (optional)
    
    Returns:
    matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean of target by category
    grouped = df.groupby(category_col)[target_col].mean().reset_index()
    
    # Create bar plot
    sns.barplot(x=category_col, y=target_col, data=grouped, ax=ax)
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Set labels
    ax.set_xlabel(category_col)
    ax.set_ylabel(f'Mean {target_col}')
    
    # Add value labels on top of the bars
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    return fig

def analyze_combined_factors(df, category1_col, category2_col, target_col, title=None):
    """
    Analyze the combined effect of two categorical variables on the target
    
    Parameters:
    df (DataFrame): The dataframe
    category1_col (str): Name of the first categorical column
    category2_col (str): Name of the second categorical column
    target_col (str): Name of the target column
    title (str): Plot title
    
    Returns:
    matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bar plot
    sns.barplot(x=category1_col, y=target_col, hue=category2_col, data=df, ax=ax)
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Set labels
    ax.set_xlabel(category1_col)
    ax.set_ylabel(f'Mean {target_col}')
    
    # Add legend
    ax.legend(title=category2_col)
    
    plt.tight_layout()
    return fig