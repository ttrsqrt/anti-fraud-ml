import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# Получаем директорию текущего скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "prepared_dataset.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "featured_dataset.csv")

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert datetime columns
    df['transdatetime'] = pd.to_datetime(df['transdatetime'])
    df['transdate'] = pd.to_datetime(df['transdate'])
    
    # Sort by client and time is crucial for lag features
    df = df.sort_values(by=['cst_dim_id', 'transdatetime'])
    return df

def create_transactional_features(df):
    print("Creating transactional features...")
    
    # Ensure transdatetime is datetime
    df['transdatetime'] = pd.to_datetime(df['transdatetime'], errors='coerce')
    
    # Remove rows with invalid datetime (crucial for rolling window)
    # initial_len = len(df)
    # df = df.dropna(subset=['transdatetime'])
    # if len(df) < initial_len:
    #     print(f"  Warning: Dropped {initial_len - len(df)} rows with invalid 'transdatetime'")
        
    # Ensure cst_dim_id is string and handle NaNs
    df['cst_dim_id'] = df['cst_dim_id'].fillna('UNKNOWN').astype(str)
    
    # 1. Time since last transaction (in seconds)
    # We need to sort by client and time for diff()
    df = df.sort_values(by=['cst_dim_id', 'transdatetime']).reset_index(drop=True)
    df['time_since_last_trans'] = df.groupby('cst_dim_id')['transdatetime'].diff().dt.total_seconds().fillna(0)
    
    # 2. Rolling statistics for Amount
    # We use rolling(on='transdatetime') which is robust
    print("  - Calculating rolling means (1d, 7d, 30d)...")
    
    # Helper to assign rolling results back to df safely
    def assign_rolling_feature(df, window, func, col_name, target_col='amount'):
        # groupby().rolling() returns a Series with MultiIndex
        # Since df is sorted by cst_dim_id and transdatetime, and groupby sorts by cst_dim_id,
        # the order of values in the result should match df exactly.
        # We use .values to ignore index alignment issues.
        if func == 'mean':
            rolled = df.groupby('cst_dim_id').rolling(window, on='transdatetime')[target_col].mean()
        elif func == 'std':
            rolled = df.groupby('cst_dim_id').rolling(window, on='transdatetime')[target_col].std()
        elif func == 'count':
            rolled = df.groupby('cst_dim_id').rolling(window, on='transdatetime')[target_col].count()
        
        # Assign by position
        df[col_name] = rolled.values
        return df

    df = assign_rolling_feature(df, '1D', 'mean', 'amount_mean_1d')
    df = assign_rolling_feature(df, '7D', 'mean', 'amount_mean_7d')
    df = assign_rolling_feature(df, '30D', 'mean', 'amount_mean_30d')
    
    # 3. Z-score (deviation from 30d mean)
    print("  - Calculating Z-scores...")
    df = assign_rolling_feature(df, '30D', 'std', 'amount_std_30d')
    df['amount_zscore_30d'] = ((df['amount'] - df['amount_mean_30d']) / (df['amount_std_30d'] + 1)).fillna(0)
    
    # 4. Count transactions in last hour
    print("  - Calculating transaction counts (1h)...")
    df = assign_rolling_feature(df, '1H', 'count', 'trans_count_1h')
    
    # 5. Velocity Features (Amount / Time since last)
    # Avoid division by zero
    print("  - Calculating velocity features...")
    df['velocity_amount_per_sec'] = df['amount'] / (df['time_since_last_trans'] + 1)
    
    # 6. Ratio Features
    print("  - Calculating ratio features...")
    df['ratio_amount_mean_1d'] = df['amount'] / (df['amount_mean_1d'] + 1)
    df['ratio_amount_mean_7d'] = df['amount'] / (df['amount_mean_7d'] + 1)
    df['ratio_amount_mean_30d'] = df['amount'] / (df['amount_mean_30d'] + 1)
    
    # 7. Cyclic Time Features
    print("  - Calculating cyclic time features...")
    # Hour (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['transdatetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['transdatetime'].dt.hour / 24)
    # Day of week (0-6)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['transdatetime'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['transdatetime'].dt.dayofweek / 7)
    
    return df

def create_behavioral_features(df):
    print("Creating behavioral features...")
    
    # 1. Device Change (compare with previous transaction of same client)
    # We assume 'last_phone_model' is the device used for *this* transaction (or the session associated with it)
    # If it's just "last seen", it might be the same for all transactions in a day.
    # But we can still check if it changed from the *previous row's* value for this client.
    
    df['prev_phone_model'] = df.groupby('cst_dim_id')['last_phone_model'].shift(1)
    df['device_changed'] = (df['last_phone_model'] != df['prev_phone_model']).astype(int)
    # First transaction is not a change
    df.loc[df['prev_phone_model'].isna(), 'device_changed'] = 0
    
    # 2. OS Change
    df['prev_os'] = df.groupby('cst_dim_id')['last_os'].shift(1)
    df['os_changed'] = (df['last_os'] != df['prev_os']).astype(int)
    df.loc[df['prev_os'].isna(), 'os_changed'] = 0
    
    # Cleanup temporary columns
    df.drop(columns=['prev_phone_model', 'prev_os'], inplace=True)
    
    return df

def create_graph_features(df):
    print("Creating graph/network features...")
    
    # 1. Receiver Popularity: How many UNIQUE senders sent money to this receiver (direction)?
    # This identifies "drop" accounts (many people sending to one).
    
    # We calculate this globally (or strictly past-only to avoid leakage, but for this task global is often acceptable baseline).
    # To be strict: "How many unique senders *up to this point*".
    # For speed, we'll do a global aggregation first, but a cumulative count is better.
    
    # Global count (simpler, slightly leaky if not careful with train/test split, but standard for "static" graph features)
    receiver_stats = df.groupby('direction')['cst_dim_id'].nunique().reset_index()
    receiver_stats.columns = ['direction', 'unique_senders_to_receiver']
    
    df = pd.merge(df, receiver_stats, on='direction', how='left')
    
    # 2. Sender Activity: How many UNIQUE receivers has this sender sent to?
    sender_stats = df.groupby('cst_dim_id')['direction'].nunique().reset_index()
    sender_stats.columns = ['cst_dim_id', 'unique_receivers_from_sender']
    
    df = pd.merge(df, sender_stats, on='cst_dim_id', how='left')
    
    return df

def main():
    df = load_data(INPUT_FILE)
    
    df = create_transactional_features(df)
    df = create_behavioral_features(df)
    df = create_graph_features(df)
    
    # Fill any NaNs created by rolling windows (though we handled some)
    df.fillna(0, inplace=True)
    
    print(f"Saving {len(df)} rows with {len(df.columns)} features to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")
    
    # Print new features
    new_cols = ['time_since_last_trans', 'amount_mean_1d', 'amount_mean_7d', 'amount_mean_30d', 
                'amount_zscore_30d', 'trans_count_1h', 'device_changed', 'os_changed', 
                'unique_senders_to_receiver', 'unique_receivers_from_sender',
                'velocity_amount_per_sec', 'ratio_amount_mean_30d', 'hour_sin', 'hour_cos']
    
    print("\nSample of new features:")
    print(df[new_cols].head().to_string())

if __name__ == "__main__":
    main()
