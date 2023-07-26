import pandas as pd


def load_csv(csv_buffer):
    df = pd.read_csv(csv_buffer)
    return df
