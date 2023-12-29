import numpy as np
import pandas as pd

def get_within_distance_matrix(x:np.ndarray, y:np.ndarray, dis:float):
    """
    Get adjacency matrix based on distance.
    If two points i,j are close to each other (within `dis`), then close(i,j) = 1
    """
    stacked = np.stack([x,y], axis=-1)

    # stacked = (N x 2)
    # stacked[None, :] = (1 x N x 2)
    # stacked[None, :] - stack = (N x N x 2) - (N x N x 2)
    close = np.linalg.norm(stacked[np.newaxis,:] - stacked[:, np.newaxis], axis= -1) < dis
    np.fill_diagonal(close, False)
    return close



def get_around_point(df:pd.DataFrame, x:np.ndarray, y:np.ndarray, dis:float):
    """
    For each data in df, count how many points are within dis, based on x and y coords of target objects.

    Assume we have M data in x and y, N data in df
    """
    stacked = np.stack([x, y], axis=-1)                     # N * 2
    sources = df[['x', 'y']].to_numpy()                     # N * 2
            # (N * 1 * 2) - (N * 2)
    close = np.linalg.norm(                                 # B * Y
                    sources[:, np.newaxis, :] - stacked
            ,axis= -1) < dis

    return np.sum(close, axis=0)



def get_agg_around_point(df:pd.DataFrame, x:np.ndarray, y:np.ndarray, dis:float):
    stacked = np.stack([x, y], axis=-1)
    bus_np = df[['x', 'y']].to_numpy()
    close = np.linalg.norm(                                 # B * Y
                    bus_np[:, np.newaxis, :] - stacked
            ,axis= -1) < dis
    num_buses = df['Bus_number'].to_numpy()[:, np.newaxis]      # B * 1

    return np.sum(num_buses * close, axis=0)