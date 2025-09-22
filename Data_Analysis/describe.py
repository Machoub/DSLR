import pandas as pd
import sys

def statistics(df):
    """Generate descriptive statistics of a DataFrame."""
    df_dropped = df.drop(df.columns[0:6], axis=1)
    stats_dict = {}
    for col in df_dropped:
        # Crée une liste propre sans les NaN
        clean_data = [x for x in df_dropped[col] if x == x]
        clean_data.sort() # Trie les données pour trouver Q1, Q2, Q3
        count_val = len(clean_data)
        mean_val = sum(clean_data) / count_val
        std_val = (sum((x - mean_val) ** 2 for x in clean_data) / count_val) ** 0.5
        min_val = clean_data[0]
        q1_val = clean_data[int(0.25 * (count_val - 1))]
        q2_val = clean_data[int(0.5 * (count_val - 1))]
        q3_val = clean_data[int(0.75 * (count_val - 1))]
        max_val = clean_data[-1]
        stats_dict[col] = [count_val, mean_val, std_val, min_val, q1_val, q2_val, q3_val, max_val]
        new_dataframe = pd.DataFrame(stats_dict, index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])

    print(new_dataframe)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <path_to_dataset>")
        sys.exit(1)
    
    try:
        df = pd.read_csv(sys.argv[1])
        statistics(df)
    except FileNotFoundError:
        print(f"Error: File not found at {sys.argv[1]}")
    except Exception as e:
        print(f"An error occurred: {e}")