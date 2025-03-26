import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


class TreeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that:
      1) Fills missing values for 'steward' and 'guards' with 'Unknown'
      2) Derives 'is_top_species' from top 10 species
      3) Bins 'tree_dbh' into categories
      4) Calculates 'problem_count'
      5) Drops columns that won't be used (spc_common)
    """
    def __init__(
        self,
        top_species_=None,
        dbh_bins=[0, 6, 12, 24, 36, 60, 1000],
        dbh_labels=['Tiny', 'Small', 'Medium', 'Large', 'XL', 'XXL'],
        problem_cols=[
            'root_stone', 'root_grate', 'root_other',
            'trunk_wire', 'trnk_light', 'trnk_other',
            'brch_light', 'brch_shoe', 'brch_other'
        ]
    ):
        self.top_species_ = top_species_
        self.dbh_bins = dbh_bins
        self.dbh_labels = dbh_labels
        self.problem_cols = problem_cols

    def fit(self, X, y=None):
        # Compute top 10 species based on the 'spc_common' frequency in *training* data, only during fit.
        top_10_species = (
            X['spc_common']
            .value_counts(dropna=True)
            .head(10)
            .index
            .tolist()
        )
        self.top_species_ = top_10_species
        return self

    def transform(self, X):
        X = X.copy()

        X['steward'] = X['steward'].fillna('Unknown')
        X['guards'] = X['guards'].fillna('Unknown')

        def mark_top_species(spc):
            if spc in self.top_species_:
                return 'Top'
            return 'Other'

        X['is_top_species'] = X['spc_common'].apply(mark_top_species)

        X['dbh_category'] = pd.cut(
            X['tree_dbh'],
            bins=self.dbh_bins,
            labels=self.dbh_labels,
            include_lowest=True
        )

        X['problem_count'] = X[self.problem_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

        X.drop(columns=['spc_common'], inplace=True, errors='ignore')

        return X


class ColumnTransformerToDataFrame(BaseEstimator, TransformerMixin):
    """
    A wrapper that takes a fitted ColumnTransformer (or any Transformer)
    and returns a DataFrame instead of a NumPy array.
    """
    def __init__(self, transformer):
        self.transformer = transformer
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        self.feature_names_ = self.transformer.get_feature_names_out()

        return self

    def transform(self, X, y=None):
        X_trans = self.transformer.transform(X)

        return pd.DataFrame(X_trans, columns=self.feature_names_, index=X.index)


# 1) Numeric columns (to be scaled)
NUMERIC_COLUMNS = [
    'tree_dbh',
    'latitude',
    'longitude',
    'problem_count'
]


# 2) Ordinal columns w/ custom categories
ORDINAL_COLUMNS = ['steward', 'dbh_category']
STEWARD_ORDER = ['Unknown', '1or2', '3or4', '4orMore']
DBH_ORDER = ['Tiny', 'Small', 'Medium', 'Large', 'XL', 'XXL']


ORDINAL_TRANSFORMER = OrdinalEncoder(
    categories=[STEWARD_ORDER, DBH_ORDER]
)


# 3) Simple Label-Encodable columns
LABEL_COLUMNS = [
    'curb_loc', 'is_top_species', 'sidewalk',
    'root_stone', 'root_grate', 'root_other',
    'trunk_wire', 'trnk_light', 'trnk_other',
    'brch_light', 'brch_shoe', 'brch_other'
]


LABEL_TRANSFORMER = OrdinalEncoder()


# 4) OneHot-encoded columns
ONEHOT_COLUMNS = ['guards', 'user_type', 'borough']
ONEHOT_TRANSFORMER = OneHotEncoder(drop='first', sparse_output=False)


# 5) Build the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_COLUMNS),
        ('ord', ORDINAL_TRANSFORMER, ORDINAL_COLUMNS),
        ('lbl', LABEL_TRANSFORMER, LABEL_COLUMNS),
        ('ohe', ONEHOT_TRANSFORMER, ONEHOT_COLUMNS),
    ],
    remainder='drop'
)
