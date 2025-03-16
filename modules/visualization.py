import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def create_correlation_heatmap(df):
    """
    Create a correlation heatmap for numerical features
    """
    # Prepare numerical columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=True,
        mask=mask,
        cmap='coolwarm',
        vmin=-1, 
        vmax=1, 
        fmt='.2f',
        linewidths=0.5,
        ax=ax
    )
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    return fig

def plot_survival_correlations(df):
    """
    Create horizontal bar chart showing correlations with survival
    """
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if 'Survived' in df.columns:
        # Calculate correlation with target
        target_corr = numeric_df.corrwith(df['Survived']).sort_values(ascending=False)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        target_corr.drop('Survived', errors='ignore').plot(
            kind='barh', 
            color=target_corr.map(lambda x: 'green' if x > 0 else 'red'),
            ax=ax
        )
        plt.title('Correlation with Survival', fontsize=16)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        return fig
    return None

def create_pairplot(df, columns, hue_col=None):
    """
    Create a pairplot for selected features
    """
    if hue_col and hue_col in columns:
        columns.remove(hue_col)
        fig = sns.pairplot(df[columns + [hue_col]], hue=hue_col, height=2.5)
    else:
        fig = sns.pairplot(df[columns], height=2.5)
    
    plt.tight_layout()
    return fig

def plot_categorical_survival_analysis(df, categorical_col):
    """
    Create heatmap and countplot for categorical variable vs survival
    """
    # Create crosstab
    crosstab = pd.crosstab(
        df[categorical_col], 
        df['Survived'],
        normalize='index'
    )
    
    # Generate heatmap
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        crosstab, 
        annot=True, 
        cmap='Blues', 
        fmt='.1%',
        cbar_kws={'label': 'Survival Rate'},
        ax=ax1
    )
    plt.title(f'Survival Rate by {categorical_col}', fontsize=16)
    plt.ylabel(categorical_col)
    plt.xlabel('Survived')
    ax1.set_xticklabels(['No', 'Yes'])
    
    # Create count plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(x=categorical_col, hue='Survived', data=df, ax=ax2)
    plt.title(f'Count by {categorical_col} and Survival Status', fontsize=16)
    plt.legend(title='Survived', labels=['No', 'Yes'])
    
    return fig1, fig2

def plot_distribution(df, column, kde=True, title=None):
    """
    Plot distribution of a column
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        sns.countplot(x=column, data=df, ax=ax)
    else:
        sns.histplot(df[column], kde=kde, ax=ax)
    
    plt.title(title or f'Distribution of {column}')
    return fig

def plot_age_speed_heatmap(df):
    """
    Create a heatmap showing survival rates by age group and speed group
    """
    # Create the crosstab
    age_speed_survival = pd.crosstab(
        [df['Age_Group']], 
        [df['Speed_Group']],
        values=df['Survived'],
        aggfunc='mean'
    ).fillna(0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        age_speed_survival,
        annot=True,
        cmap='YlGnBu',
        fmt='.1%',
        cbar_kws={'label': 'Survival Rate'},
        ax=ax
    )
    plt.title('Survival Rate by Age Group and Speed Group', fontsize=16)
    plt.tight_layout()
    return fig

def plot_grouped_survival(df, x_col, hue_col=None, title=None):
    """
    Plot bar chart showing survival rate by group
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if hue_col:
        sns.barplot(x=x_col, y='Survived', hue=hue_col, data=df, ax=ax)
    else:
        sns.barplot(x=x_col, y='Survived', data=df, estimator=np.mean, ax=ax)
    
    plt.title(title or f'Survival Rate by {x_col}')
    plt.ylabel('Survival Rate')
    return fig

def plot_feature_engineering_impact(df, target_col='Survived'):
    """
    Plot the impact of engineered features on the target variable
    
    Parameters:
    df (DataFrame): The preprocessed dataframe
    target_col (str): The target column name
    
    Returns:
    matplotlib.figure.Figure: Figure with subplots showing feature impact
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Age Group impact
    if 'Age_Group' in df.columns:
        survival_by_age = df.groupby('Age_Group')[target_col].mean().reset_index()
        sns.barplot(x='Age_Group', y=target_col, data=survival_by_age, ax=axes[0, 0])
        axes[0, 0].set_title('Survival Rate by Age Group', fontsize=14)
        axes[0, 0].set_ylabel('Survival Rate')
        
        # Add value labels
        for p in axes[0, 0].patches:
            axes[0, 0].annotate(f'{p.get_height():.2f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'bottom')
    
    # Speed Group impact
    if 'Speed_Group' in df.columns:
        survival_by_speed = df.groupby('Speed_Group')[target_col].mean().reset_index()
        sns.barplot(x='Speed_Group', y=target_col, data=survival_by_speed, ax=axes[0, 1])
        axes[0, 1].set_title('Survival Rate by Speed Group', fontsize=14)
        axes[0, 1].set_ylabel('Survival Rate')
        
        # Add value labels
        for p in axes[0, 1].patches:
            axes[0, 1].annotate(f'{p.get_height():.2f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'bottom')
    
    # Safety Score impact
    if 'Safety_Score' in df.columns:
        survival_by_safety = df.groupby('Safety_Score')[target_col].mean().reset_index()
        sns.barplot(x='Safety_Score', y=target_col, data=survival_by_safety, ax=axes[1, 0])
        axes[1, 0].set_title('Survival Rate by Safety Score', fontsize=14)
        axes[1, 0].set_xlabel('Safety Score (Number of Safety Features Used)')
        axes[1, 0].set_ylabel('Survival Rate')
        axes[1, 0].set_xticks([0, 1, 2])
        axes[1, 0].set_xticklabels(['0 (None)', '1 (Either)', '2 (Both)'])
        
        # Add value labels
        for p in axes[1, 0].patches:
            axes[1, 0].annotate(f'{p.get_height():.2f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'bottom')
    
    # Age/Speed Ratio Bin impact
    if 'Age_Speed_Ratio_Bin' in df.columns:
        survival_by_ratio = df.groupby('Age_Speed_Ratio_Bin')[target_col].mean().reset_index()
        sns.barplot(x='Age_Speed_Ratio_Bin', y=target_col, data=survival_by_ratio, ax=axes[1, 1])
        axes[1, 1].set_title('Survival Rate by Age/Speed Ratio', fontsize=14)
        axes[1, 1].set_ylabel('Survival Rate')
        
        # Add value labels
        for p in axes[1, 1].patches:
            axes[1, 1].annotate(f'{p.get_height():.2f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha = 'center', va = 'bottom')
    
    plt.tight_layout()
    return fig