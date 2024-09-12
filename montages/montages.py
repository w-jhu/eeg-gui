import numpy as np
import pandas as pd

def generate_bipolar_montages(dataframe):
    bi_pairs = [('Fp1', 'F7'),
                ('F7', 'T3'),
                ('T3', 'T5'),
                ('T5', 'O1'),
                ('Fp2', 'F8'),
                ('F8', 'T4'),
                ('T4', 'T6'),
                ('T6', 'O2'),
                ('Fp1', 'F3'),
                ('F3', 'C3'),
                ('C3', 'P3'),
                ('P3', 'O1'),
                ('Fp2', 'F4'),
                ('F4', 'C4'),
                ('C4', 'P4'),
                ('P4', 'O2'),
                ('Fz', 'Cz')]

    bi_values = []
    bi_labels = []

    for first, second in bi_pairs:
        if first in dataframe.columns and second in dataframe.columns:
            if not dataframe[first].isnull().any() and not dataframe[second].isnull().any():
                montage_values = dataframe[first] - dataframe[second]
                bi_values.append(montage_values)
            bi_labels.append(f"{first}-{second}")

    bi_values = pd.DataFrame(bi_values).T
    bi_labels = pd.Series(bi_labels)
    bi_values.rename(columns=dict(zip(bi_values.columns, bi_labels)), inplace=True)

    return bi_values

def car_montage(df, which_chs):
    car_labels = []
    car_values = df.copy()

    valid_columns = [col for col in df.columns if df[col].notna().all()]
    invalid_columns = [col for col in df.columns if col not in valid_columns]
    
    valid_chs = list(set(which_chs).intersection(valid_columns))
    valid_chs_indices = [df.columns.get_loc(channel) for channel in valid_chs]
    
    average = np.nanmean(car_values.iloc[:, valid_chs_indices], axis=1, keepdims=True)

    car_values -= average

    for invalid_col in invalid_columns:
        car_values[invalid_col] = None
        
    for label in car_values.columns:
        car_labels.append(label + '-CAR')

    car_values.rename(columns=dict(zip(car_values.columns, car_labels)), inplace=True)
    return car_values