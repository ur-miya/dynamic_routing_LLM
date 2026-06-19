import pandas as pd

def sample_train_data(train_path="data/raw/oasst1/train.csv", n_samples=1000, random_state=42):
    """
    Loading train.csv and return random subset of n_samples.
    """
    df = pd.read_csv(train_path)
    if n_samples > len(df):
        print(f"Required {n_samples}, train has {len(df)}. Everyone is used")
        return df
    sampled = df.sample(n=n_samples, random_state=random_state)
    return sampled