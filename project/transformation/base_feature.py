from abc import ABC, abstractmethod
from typing import TypeAlias, Callable, Self
import pandas as pd

DF: TypeAlias = pd.DataFrame
Pipeline: TypeAlias = Callable[[DF], DF]



class FeatureBaseClass(ABC):

    @property
    @abstractmethod
    def list_required_columns(self) -> list[str]:
        ...

    def df_validated(self, df:DF) -> bool:
        return all(col in df.columns for col in self.list_required_columns)

    @abstractmethod
    def transform(self, df:DF) -> DF:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__

class FeaturePipe:
    def __init__(self, feature_list:list[FeatureBaseClass]):
        self.feature_list = feature_list

    def append_feature(self, feature:FeatureBaseClass):
        self.feature_list.append(feature)

    def __add__(self, features:FeatureBaseClass|list[FeatureBaseClass]|Self):
        if isinstance(features, FeatureBaseClass):
            self.feature_list.append(features)
        elif isinstance(features, list):
            if not all(isinstance(f, FeatureBaseClass) for f in features):
                raise ValueError("All elements should be FeatureBaseClass")
        elif isinstance(features, Self):
            self.feature_list += features.feature_list

        else:
            raise TypeError("features should be either a FeatureBaseClass, list of FeatureBaseClass, or another FeaturePipe element.")

    def build(self, df:DF):
        df_ = df.copy()
        for i, feature in enumerate(self.feature_list):
            if not feature.df_validated(df_):
                raise ValidationError(f"Feature chain error in step {i} -> {feature}. Need {feature.list_required_columns}!")
            df_ = feature.transform(df_)
        self._df = df_

        return self._df

    def show_features(self):
        for i, f in enumerate(self.feature_list):
            print(i, f, sep='\t')
        print("Total columns: ", len(self._df.columns))

    @property
    def df(self):
        return self._df

class ValidationError(BaseException):
    pass