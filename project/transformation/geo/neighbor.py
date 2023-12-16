import numpy as np

def get_within_distance_matrix(x:np.ndarray, y:np.ndarray, dis:float):
    stacked = np.stack([x,y], axis=-1)

    # stacked = (N x 2)
    # stacked[None, :] = (1 x N x 2)
    # stacked[None, :] - stack = (N x N x 2) - (N x N x 2)
    close = np.linalg.norm(stacked[np.newaxis,:] - stacked[:, np.newaxis], axis= -1) < dis
    np.fill_diagonal(close, False)
    return close