import numpy as np
import pandas as pd

# A function to assign a category based on the madrs score
def assign_category(score):
    if score >= 0 and score <= 6:
        return "absent"
    elif score >= 7 and score <= 19:
        return "mild depression"
    elif score >= 20 and score <= 34:
        return "moderate depression"
    elif score >= 35 and score <= 60:
        return "severe depression"
    else:
        return "invalid score"

# Creates a new column with 1.0 for weekend dates and 0.0 for weekdays
def is_weekend(df):
    df.date = pd.to_datetime(df.timestamp, infer_datetime_format=True)
    df.loc[:, "is_weekend"] = df.timestamp.dt.dayofweek  # returns 0-4 for Monday-Friday and 5-6 for Weekend
    df.loc[:, 'is_weekend'] = df['is_weekend'].apply(lambda d: 1.0 if d > 4 else 0.0)

    return df

def sin_transform(values):
    """
    Applies SIN transform to a series value.
    Args:
        values (pd.Series): A series to apply SIN transform on.
    Returns
        (pd.Series): The transformed series.
    """

    return np.sin(2 * np.pi * values / len(set(values)))


def cos_transform(values):
    """
    Applies COS transform to a series value.
    Args:
        values (pd.Series): A series to apply SIN transform on.
    Returns
        (pd.Series): The transformed series.
    """
    return np.cos(2 * np.pi * values / len(set(values)))


def date_engineering(data):
    # Ensure timestamp is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Extract date components
    data["year"] = data["timestamp"].dt.year
    data["month"] = data["timestamp"].dt.month
    data["weekday"] = data["timestamp"].dt.weekday
    data["week"] = data["timestamp"].dt.isocalendar().week
    data["day"] = data["timestamp"].dt.day

    # Apply sin and cos transforms
    data["month_sin"] = sin_transform(data["month"])
    data["weekday_sin"] = sin_transform(data["weekday"])
    data["week_sin"] = sin_transform(data["week"])
    data["day_sin"] = sin_transform(data["day"])

    data["month_cos"] = cos_transform(data["month"])
    data["weekday_cos"] = cos_transform(data["weekday"])
    data["week_cos"] = cos_transform(data["week"])
    data["day_cos"] = cos_transform(data["day"])

    # Drop original date components
    data = data.drop(columns=['year', 'month', 'weekday', 'week', 'day'])

    return data

def one_hot_encoding(df):
    # edu encoding
    df['edu'].replace(to_replace=['', '6-10', '11-15', '16-20'], value=[0, 1, 2, 3], inplace=True)
    # category_madrs
    df['category_madrs'].replace(to_replace=['moderate depression', 'mild depression'], value=[0, 1], inplace=True)
    #age
    df['age'].replace(to_replace=['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69'],
                      value=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], inplace=True)
    return df

# adds activity quantile
def add_activity_quantile(df):
    df = df.astype({"user_id": str})
    ids = list(np.unique((df[['user_id']])))

    df["activity_quantile"] = pd.qcut(df["activity"].rank(method='first'), [0, .25, .75, 1], labels=["low", "medium", "high"])
    df['activity_quantile'].replace(to_replace=['low', 'medium', 'high'], value=[0, 1, 2], inplace=True)

    d = pd.DataFrame()
    for user in ids:
        user_df = df[(df["user_id"] == user)]
        user_df["user_activity_quantile"] = pd.qcut(user_df["activity"].rank(method='first'), [0, .25, .75, 1],
                                                  labels=[0, 1, 2])
        d = pd.concat([d, user_df])
    df = d

    return df