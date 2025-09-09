import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the input Excel file.
    
    Args:
        file_path: Path to the input Excel file
        
    Returns:
        Preprocessed DataFrame with proper data types
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Convert OrderDate to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['OrderDate']):
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    
    # Ensure required columns exist
    required_columns = ['OrderDate', 'CustomerPhone', 'CustomerName', 'Product', 'Revenue']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Sort by CustomerPhone and OrderDate for easier processing
    df = df.sort_values(['CustomerPhone', 'OrderDate'])
    
    return df


def identify_target_customers(df: pd.DataFrame) -> List[str]:
    """
    Identify customers who ordered frequently in Q1 but churned afterward.
    
    Args:
        df: Preprocessed DataFrame with transaction data
        
    Returns:
        List of CustomerPhone values that meet the criteria
    """
    # Get the latest year in the dataset
    latest_year = df['OrderDate'].dt.year.max()
    
    # Define date ranges
    q1_start = datetime(latest_year, 1, 1)
    q1_end = datetime(latest_year, 3, 31)
    churn_start = datetime(latest_year, 4, 1)
    churn_end = datetime(latest_year, 12, 31)
    
    # Filter Q1 orders
    q1_orders = df[df['OrderDate'].between(q1_start, q1_end)]
    
    # Get customers with ≥3 orders in Q1
    q1_customers = q1_orders.groupby('CustomerPhone').filter(lambda x: len(x) >= 3)
    frequent_q1_customers = q1_customers['CustomerPhone'].unique()
    
    # Filter churn period orders
    churn_orders = df[df['OrderDate'].between(churn_start, churn_end)]
    
    # Find customers who didn't order during churn period
    churned_customers = [
        phone for phone in frequent_q1_customers 
        if phone not in churn_orders['CustomerPhone'].unique()
    ]
    
    return churned_customers


def calculate_order_metrics(customer_orders: pd.DataFrame) -> Dict:
    """
    Calculate order frequency and gap metrics for a customer.
    
    Args:
        customer_orders: DataFrame containing a single customer's orders
        
    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['Number of Orders'] = len(customer_orders)
    metrics['Total Revenue'] = customer_orders['Revenue'].sum()
    
    # Order dates for time-based calculations
    order_dates = customer_orders['OrderDate'].sort_values().reset_index(drop=True)
    first_order = order_dates.min()
    last_order = order_dates.max()
    
    # Calculate order frequency (orders per month)
    months_active = (last_order - first_order).days / 30.44
    metrics['Order Frequency'] = metrics['Number of Orders'] / max(months_active, 1)
    
    # Calculate average order gap (in days)
    if len(order_dates) > 1:
        gaps = (order_dates.diff().dropna()).dt.days
        metrics['Average Order Gap'] = gaps.mean()
    else:
        metrics['Average Order Gap'] = 0
    
    # Get unique products
    metrics['Product List'] = ', '.join(sorted(customer_orders['Product'].unique()))
    
    return metrics


def generate_customer_profiles(df: pd.DataFrame, target_customers: List[str]) -> pd.DataFrame:
    """
    Generate customer profiles for the target customers.
    
    Args:
        df: Complete transaction DataFrame
        target_customers: List of CustomerPhone values to generate profiles for
        
    Returns:
        DataFrame with customer profiles
    """
    profiles = []
    
    for phone in target_customers:
        # Get all orders for this customer
        customer_orders = df[df['CustomerPhone'] == phone].copy()
        
        # Get customer name (first occurrence)
        customer_name = customer_orders.iloc[0]['CustomerName']
        
        # Calculate metrics
        metrics = calculate_order_metrics(customer_orders)
        
        # Create profile
        profile = {
            'Name': customer_name,
            'Number of Orders': metrics['Number of Orders'],
            'Total Revenue': metrics['Total Revenue'],
            'Order Frequency': metrics['Order Frequency'],
            'Average Order Gap': metrics['Average Order Gap'],
            'Product List': metrics['Product List']
        }
        
        profiles.append(profile)
    
    # Convert to DataFrame and format
    profile_df = pd.DataFrame(profiles)
    
    # Format currency and round numbers
    profile_df['Total Revenue'] = profile_df['Total Revenue'].apply(
        lambda x: f"${x:,.2f}" if pd.notnull(x) else "$0.00"
    )
    
    # Round numeric columns
    numeric_cols = ['Order Frequency', 'Average Order Gap']
    profile_df[numeric_cols] = profile_df[numeric_cols].round(2)
    
    return profile_df


def main():
    # Configuration
    input_file = 'customer_orders.xlsx'  # Update with your input file name
    output_file = 'customer_profiles.xlsx'
    
    try:
        # Load and preprocess data
        print(f"Loading data from {input_file}...")
        df = load_data(input_file)
        
        # Identify target customers
        print("Identifying target customers...")
        target_customers = identify_target_customers(df)
        
        if not target_customers:
            print("No customers matched the criteria.")
            return
            
        print(f"Found {len(target_customers)} customers who ordered frequently in Q1 but churned afterward.")
        
        # Generate and save customer profiles
        print("Generating customer profiles...")
        profiles = generate_customer_profiles(df, target_customers)
        
        # Save to Excel
        profiles.to_excel(output_file, index=False)
        print(f"Customer profiles saved to {output_file}")
        
        # Display summary
        print("\nSummary of customer profiles:")
        print(f"- Total customers: {len(profiles)}")
        print(f"- Average orders per customer: {profiles['Number of Orders'].mean():.1f}")
        print(f"- Average revenue per customer: ${profiles['Total Revenue'].str.replace('$', '').str.replace(',', '').astype(float).mean():,.2f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
