from transformation.base_feature import DF
from transformation import FeatureBaseClass, ValidationError
from transformation.geo import tw_transformer

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

class XYCoord(FeatureBaseClass):
    """
    Adds Cartesian coordinate system features based on latitude and longitude columns.

    This class is designed to transform a DataFrame by adding Cartesian coordinates (x and y) based on latitude and longitude columns.

    Parameters:
    - lat_col (str): The name of the latitude column in the DataFrame. Default is 'lat'.
    - lng_col (str): The name of the longitude column in the DataFrame. Default is 'lng'.
    - geo_transformer: A transformer function that takes latitude and longitude as input and returns Cartesian coordinates (x, y).

    Attributes:
    - geo_transformer_func: A lambda function that applies the geo_transformer to latitude and longitude.
    - lat_col (str): The name of the latitude column.
    - lng_col (str): The name of the longitude column.

    Example:
    ```python
    # Create an instance of XYCoord with default parameters
    xy_transformer = XYCoord()

    # Transform a DataFrame containing 'lat' and 'lng' columns
    transformed_df = xy_transformer.transform(input_df)
    ```

    Note: Ensure that the input DataFrame has the required 'lat' and 'lng' columns for transformation.
    """
    def __init__(self, lat_col:str = 'lat', lng_col:str = 'lng', geo_transformer = tw_transformer):
        self.geo_transformer_func = lambda x,y: geo_transformer.transform(x,y)
        self.lat_col = lat_col
        self.lng_col = lng_col

    @property
    def list_required_columns(self):
        return ["lat", "lng"]

    def transform(self, df:DF) -> DF:
        if {'x', 'y'} in set(df.columns):
            return df
        x,y = self.geo_transformer_func(df[self.lat_col], df[self.lng_col])
        df['x'] = x
        df['y'] = y
        return df


class FeatureFromStation(FeatureBaseClass):
    """
    Intermediate abstract class for features that require x,y coordinates, name, and address for stations.
    """
    @property
    def list_required_columns(self):
        return ['x','y', 'address']

    @property
    def geo_transformer(self):
        return tw_transformer

    @staticmethod
    def get_around_point(df:DF, x:np.ndarray, y:np.ndarray, dis:float) -> np.ndarray:
        """
        For each data in df, count how many points are within dis, based on x and y coords of target objects.

        Assume we have M data in x and y, N data in df
        """
        stacked = np.stack([x, y], axis=-1)                     # M * 2
        sources = df[['x', 'y']].to_numpy()                     # N * 2 for youbike station
                # (N * 1 * 2) - (M * 2)
        close = np.linalg.norm(                                 # N * M * 2 -> # N * M
                        sources[:, np.newaxis, :] - stacked
                ,axis= -1) < dis

        return np.sum(close, axis=1)

    def __repr__(self) -> str:
        if hasattr(self, 'dist'):
            return super().__repr__() + f"[distance = {self.dist}]"
        else:
            return super().__repr__()


    @staticmethod
    def get_A_close_to_B(coordsA:np.ndarray, coordsB:np.ndarray, dist:float):
        kd1 = KDTree(coordsA)
        kd2 = KDTree(coordsB)

        close = np.sum(
            kd1.sparse_distance_matrix(kd2, dist).toarray()
            , axis= -1) > 0
        return close


class ConvenientStoreFeature(FeatureFromStation):
    def __init__(self, convenient_df:DF, dist:float = 300):
        """
        `convenient_df` should be a df with at least two columns of coords for each conveneint store.
        Changed to x,y using tw_transformer during init
        """
        self.conv_df = convenient_df
        self.dist = dist
        if not all(col in ['x', 'y'] for col in convenient_df.columns):
            x,y = self.geo_transformer.transform(self.conv_df['lat'], self.conv_df['lng'])
            self.conv_df['x'] = x
            self.conv_df['y'] = y

    def transform(self, df: DF) -> DF:
        """
        For each data in df, calculate the number of nearest convenient stores within a distance.
        """
        conv_count = self.get_around_point(df, self.conv_df['x'], self.conv_df['y'], dis=self.dist)
        df['conv_count'] = conv_count
        return df

class VillageFeature(FeatureFromStation):
    def __init__(self, village_data:DF, village_coord:DF, village_name_col:tuple[str] = ('里別', 'VILLNAME')):
        """
        Village data should be in the following form:
        village_name, lat(x), lng(y), [...features]
        """
        if not any(col in ['x', 'y'] for col in village_coord.columns):
            try:
                x,y = self.geo_transformer.transform(village_coord['lat'], village_coord['lng'])
                village_coord['x'] = x
                village_coord['y'] = y
            except Exception as e:
                raise ValidationError("Village data must have coordinate") from e

        # merge data
        self.village_data =( village_data.merge(village_coord, left_on=village_name_col[0], right_on=village_name_col[1])
                                .drop(columns=list(village_name_col))
                            )

    def transform(self, df: DF) -> DF:
        """
        For each village in `self.village_data`, calculate weight for each
        YouBike station in `df`. Weight is given by inverse distance. If
        a YouBike station is further from the village's coordinate, it is
        assigned less weight from the village.

        For example, if station A is 2km away from village X, 5km away from
        village Y, then the weight is given by 0.5 to X, and 0.2 to Y.
        Assign these to an M x N array `self.weight`.

        Each village has K features such as income for each age, and for each
        YouBike Station the corresponding feature is the weighted average of all
        villages.

        Parameters:
        - df (DF): DataFrame containing YouBike station data.

        Returns:
        - DF: Transformed DataFrame with added weight columns.
        """
        self.weights = self._calculate_weights(df)
        village_features_cols = [col for col in self.village_data.columns if col not in ['x', 'y']]

        village_features = self.village_data[village_features_cols].values  # M x K
        #self.weights.T                                                     # M x N -> N x M
        weighted_sum = self.weights.T@village_features                      # N x K

        return pd.concat([df, pd.DataFrame(weighted_sum, columns = village_features_cols)], axis=1)



        ## village_data excluding 'x', 'y' has K columns

    def _calculate_weights(self, df:DF):

        # Extract coordinates from the DataFrame
        village_coordinates = self.village_data[['x', 'y']].values          # M x 2
        station_coordinates = df[['x', 'y']].values                         # N x 2

        # Calculate Euclidean distances between all village-station pairs

        distances = np.linalg.norm(
                village_coordinates[:, np.newaxis, :]                       # M x 1 x 2
            -   station_coordinates,                                        #     N x 2
            axis=2)                                                         # M x N

        # Assign weight as the inverse of distance
        weights = 1.0 / distances

        # Normalize weights to sum to 1 for each YouBike
        weights /= np.sum(weights, axis=0)                                  # M x N

        return weights

class InAlley(FeatureBaseClass):
    """
    Whether an YouBike station is in alley or not.
    """
    @property
    def list_required_columns(self) -> list[str]:
        return ["address"]

    def transform(self, df: DF) -> DF:
        df['in_alley'] = df.address.str.contains("巷", regex=False)
        return df

class BusAggregateFeature(FeatureFromStation):
    def __init__(self, bus_df:DF, dist:float = 400, bus_num_col:str='Bus_number'):
        """
        Add total number of routes of buses around a given coordinate, within `dist` meters.

        Parameters
        ---
        bus_df: dataframe with each row representing a bus stop. Must have lat,lng or x, y, and a columns with number of bus.
        """
        self.bus_df = bus_df
        self.bus_num_col = bus_num_col
        self.dist = dist

    def transform(self, df: DF) -> DF:
        df['bus_arround'] = self.get_agg_bus_around_point(df)
        return df

    ## to do
    def get_agg_bus_around_point(self, df:DF):
        """
        df is the one to pass in (the youbike data), (x, y) is the arrays of bus data.
        """
        stacked = df[['x', 'y']].to_numpy()                         # Y x 2
        bus_np = self.bus_df[['x', 'y']].to_numpy()                 # B x 2
        close = np.linalg.norm(                                     # B x Y
                        bus_np[:, np.newaxis, :] -                  # B x 1 x 2
                        stacked                                     #     Y x 2
                ,axis= -1) < self.dist                              # B x Y (Bool)
        num_buses = self.bus_df[
            self.bus_num_col].to_numpy()[:, np.newaxis]             # B x 1

        return np.sum(num_buses * close, axis=0)                    # B x Y -> Y

class BikeRouteFeaure(FeatureFromStation):
    def __init__(self, bike_route_df:DF, dist:float = 1000.0):
        if not any(col in ['x', 'y'] for col in bike_route_df.columns):
            try:
                x,y = self.geo_transformer.transform(bike_route_df['lat'], bike_route_df['lng'])
                bike_route_df['x'] = x
                bike_route_df['y'] = y
            except Exception as e:
                raise ValidationError("Bike route data must have coordinate") from e
        self.bike_route  = bike_route_df[['x','y']]
        self.dist = dist

    def transform(self, df: DF) -> DF:
        # For this, I will try use scipy KDTree since there are too many coordinates.
        df['around_riverside'] = self.get_A_close_to_B(
                                        df[['x','y']],
                                        self.bike_route[['x', 'y']],
                                        self.dist
                                        )
        return df

class HighSchoolFeature(FeatureFromStation):
    def __init__(self, highschool_df:DF, dist:float = 1000.0):
        if not any(col in ['x', 'y'] for col in highschool_df.columns):
            try:
                x,y = self.geo_transformer.transform(highschool_df['lat'], highschool_df['lng'])
                highschool_df['x'] = x
                highschool_df['y'] = y
            except Exception as e:
                raise ValidationError("Highschool data must have coordinate") from e
        self.hs_position  = highschool_df[['x','y']]
        self.dist = dist

    def transform(self, df: DF) -> DF:
        df['around_highschool'] = self.get_A_close_to_B(
                                        df[['x', 'y']],
                                        self.hs_position[['x', 'y']],
                                        self.dist
                                        )
        return df
    # def


class FeaturesSpaceTime(FeatureBaseClass):

    @property
    def list_required_columns(self) ->list[str]:
        return ['x', 'y', 'time']

    @property
    def geo_transformer(self):
        return tw_transformer

class MRTIndex(FeatureFromStation):
    def __init__(self, mrt_df:DF, dist:float = 200):
        """
        Preprocessing for MRTTimeFueature. Adds nearest MRT exit aa station name. Return `np.nan` if no MRT found around
        """

        if not any(col in ['x', 'y'] for col in mrt_df.columns):
            try:
                x,y = self.geo_transformer.transform(mrt_df['lat'], mrt_df['lng'])
                mrt_df['x'] = x
                mrt_df['y'] = y
            except Exception as e:
                raise ValidationError("MRT data must have coordinate") from e

        self.mrt_df = mrt_df.copy()
        self.dist = dist

    def generate_target_MRT_pair(self, df:DF) -> list[list[int]]:
        target_kdtree = KDTree(df[['x', 'y']].to_numpy())
        exit_kdtree = KDTree(self.mrt_df[['x', 'y']].to_numpy())

        list_of_neighbors = target_kdtree.query_ball_tree(exit_kdtree, self.dist)       # list of id_exits

        return [neighbor[0] if len(neighbor)>0 else None for neighbor in list_of_neighbors]

    def transform(self, df: DF) -> DF:

        neighbors = self.generate_target_MRT_pair(df)       # closest MRT exit for each df columns
        self.mrt_df.loc[-1, 'station'] = None
        neighbors  = [-1 if n is None else n for n in neighbors]

        MRT_name = self.mrt_df.loc[neighbors, 'station']
        df['MRT_name'] = list(MRT_name)
        return df

class MRTTimeFeature(FeaturesSpaceTime):

    def __init__(self, mrt_df:DF, dist:float=200):
        """
        Generates weighted number of MRT exits.
        Weight is given by log average in/out flows, where higher flows representing a higher importance
        of the MRT feature.

        For example, if two target points share the same amount of MRT exits,
        but MRT beside target A has a higher flow amount,
        then target A is assigned a higher value.

        Parameter
        ---
        mrt_df (`pd.DataFrame`): a dataframe with ['stattion', 'hour', 'in', 'out']
        dist (`float`): maximum distance to count as 'around an MRT exit'

        Explanations
        ---
        The weighted number of MRTs around is calculated by the following concept:
        1. Find all exits around a series of target point. Implemented using `KDTree`
        2. Calculate the average of all nearby MRT points.
        3. Repeat for all 24 hours for each  points
        4. Merge back to data based on station/hour index pair
        """
        self.mrt_df = mrt_df[['station', 'hour', 'in', 'out']].set_index(['station', 'hour'])
        self._inflow_dict = self.mrt_df['in'].to_dict()
        self._outflow_dict = self.mrt_df['out'].to_dict()

    @property
    def list_required_columns(self) ->list[str]:
        return super().list_required_columns + ['MRT_name']


    def retrieve_inflow(self, station, hour):
        return self._inflow_dict.get((station, hour), 0)

    def retrieve_outflow(self, station, hour):
        return self._outflow_dict.get((station, hour), 0)

    def transform(self, df: DF) -> DF:
        df['hour'] = df['time'].dt.hour


        df['MRT_inflow'] = [self.retrieve_inflow(s,h) for s,h in zip(df['MRT_name'], df['hour'])]
        df['MRT_outflow'] =[self.retrieve_outflow(s,h) for s,h in zip(df['MRT_name'], df['hour'])]

        df.drop(columns=['hour', 'MRT_name'])

        return df


class WeatherFeature(FeaturesSpaceTime):
    def __init__(self, weather_df:DF):
        """
        Add precipitation, temperature, and wind info into dataframe.

        THe weather data should cover all time series. Preprocessing on determining the closest observatory ID is necessary.
        """

        self.weather_df = weather_df

    @property
    def list_required_columns(self):
        return super().list_required_columns + ['weather_station']