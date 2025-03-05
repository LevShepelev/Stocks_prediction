from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return data_scaled, scaler
