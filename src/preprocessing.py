import pandas as pd
import pickle
from datetime import datetime
from src.load_data import INTERMEDIATE_DATA_DIR


def filter_users_items(df, user_min=5, item_min=10):
    """
    Filter users and items based on minimum review counts.

    Args:
        df (pd.DataFrame): DataFrame with at least 'reviewerID' and 'asin' columns.
        user_min (int): Minimum number of reviews a user must have to be kept.
        item_min (int): Minimum number of reviews an item must have to be kept.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only users/items meeting the thresholds.
    """
    user_counts = df['reviewerID'].value_counts()
    item_counts = df['asin'].value_counts()
    return df[df['reviewerID'].isin(user_counts[user_counts >= user_min].index) &
              df['asin'].isin(item_counts[item_counts >= item_min].index)]


def time_split(df, t1_quantile=0.7, t2_quantile=0.85):
    """
    Split DataFrame into train, validation, and test sets based on reviewTime quantiles.

    Args:
        df (pd.DataFrame): DataFrame with 'reviewTime', 'reviewerID', and 'asin' columns.
        t1_quantile (float): Quantile threshold for train/validation split.
        t2_quantile (float): Quantile threshold for validation/test split.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if not pd.api.types.is_datetime64_any_dtype(df['reviewTime']):
        df['reviewTime'] = pd.to_datetime(df['reviewTime'])

    df = df.sort_values("reviewTime")
    t1, t2 = df['reviewTime'].quantile([t1_quantile, t2_quantile])
    train = df[df['reviewTime'] <= t1]

    train_users, train_items = set(train['reviewerID']), set(train['asin'])
    val = df[(df['reviewTime'] > t1) & (df['reviewTime'] <= t2) &
            (df['reviewerID'].isin(train_users)) & (df['asin'].isin(train_items))]

    test = df[(df['reviewTime'] > t2) &
            (df['reviewerID'].isin(train_users)) & (df['asin'].isin(train_items))]

    print(f"Split sizes -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def prepare_and_save_data(df, user_min_reviews=5, item_min_reviews=10, t1_quantile=0.7, t2_quantile=0.85):
    """
    Filter users/items, split by reviewTime, and save train+val and test sets with timestamped filenames.

    Args:
        df (pd.DataFrame): Input DataFrame.
        user_min_reviews (int): Minimum reviews per user.
        item_min_reviews (int): Minimum reviews per item.
        t1_quantile (float): Quantile threshold for train/validation split.
        t2_quantile (float): Quantile threshold for validation/test split.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train, val, test = time_split(filter_users_items(df, user_min_reviews, item_min_reviews),
                                  t1_quantile, t2_quantile)

    with open(INTERMEDIATE_DATA_DIR / f"train_val_{timestamp}.pkl", "wb") as f:
        pickle.dump((train, val), f)
    with open(INTERMEDIATE_DATA_DIR / f"test_{timestamp}.pkl", "wb") as f:
        pickle.dump(test, f)

    print(f"Saved train+val and test in data/intermediate with timestamp: {timestamp}")
    return train, val, test


def expand_dict_column(df, dict_col):
    """
    Expands a dictionary column in a DataFrame into separate columns.
    Missing keys are filled with None. Non-dict entries are treated as empty dicts.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        dict_col (str): Column name containing dictionaries
    
    Returns:
        pd.DataFrame: DataFrame with the dictionary column expanded into separate columns
    """
    if dict_col not in df.columns:
        raise ValueError(f"Column '{dict_col}' not found in DataFrame")
    
    df_safe = df[dict_col].apply(lambda x: x if isinstance(x, dict) else {})
    
    all_keys = set(k for d in df_safe for k in d)
    
    dict_expanded = pd.json_normalize(df_safe).reindex(columns=all_keys, fill_value=None)
    
    df_expanded = pd.concat([df.drop(columns=[dict_col]), dict_expanded], axis=1)
    
    return df_expanded


def expand_video_games_rank(df, rank_col="rank", new_col="games_rank"):
    """
    Extracts the 'Video Games' top-level category rank from a messy rank column
    and adds it as a clean numeric column. Ignores all other categories or text.

    Args:
        df (pd.DataFrame): DataFrame containing the rank column.
        rank_col (str): Name of the column with category rank info.
        new_col (str): Name for the new column to store Video Games rank.

    Returns:
        pd.DataFrame: DataFrame with only the new Video Games rank column added.
    """

    def extract_video_game_rank(cell):
        if isinstance(cell, str):
            items = [cell]
        elif isinstance(cell, list):
            items = cell
        else:
            return None

        for item in items:
            if not isinstance(item, str):
                continue
                
            item = html.unescape(item)
            match = re.search(r'#([\d,]+) in ([^(]+)', item)
            if match:
                rank = int(match.group(1).replace(',', ''))
                category = match.group(2).split('>')[0].strip()
                if category.lower() == "video games":
                    return rank
        return None

    df[new_col] = df[rank_col].apply(extract_video_game_rank)

    return df