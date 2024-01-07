import numpy as np
import pandas as pd


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