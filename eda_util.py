import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from typing import Any

class SaveLoad:
    def pickle_save(obj: Any, file_path: str) -> None:
        print('Pickle saving...')
        with open(file_path, 'wb') as fp:
            pickle.dump(obj, fp)
            print('Pickle saved.')
        return
    
    def pickle_load(file_path: str) -> Any:
        print('Pickle loading...')
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
        return obj
    
class DataImport:
    def split_data(data, random_state=1048576):
        train, test = train_test_split(data, test_size=0.2, random_state=random_state)
        val, test = train_test_split(test, test_size=0.5, random_state=random_state)
        return train, val, test
    
    def get_train_test(path: str, index_col: int = None, target: str = None, test_size: float = 0.1, random_state: int = 1048576, verbose: int = 1) -> tuple[pd.DataFrame]:
        train_csv = pd.read_csv(path, index_col=index_col)
        X, y = FeatureTarget.feature_target_split(train_csv, target=target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        if verbose > 0:
            print(f'Shape of X_train is: {X_train.shape}; shape of y_train is: {y_train.shape}')
            print(f'Shape of X_test is: {X_test.shape}; shape of y_test is: {y_test.shape}')
        return X_train, X_test, y_train, y_test
    
    def datasplit_info(train_df, val_df, test_df):
        columns = []
        train_nulls_cnt = []
        val_nulls_cnt = []
        test_nulls_cnt = []
        dtypes = []

        for col in train_df.columns:
            columns.append(col)
            train_nulls_cnt.append(train_df[col].isna().sum())
            val_nulls_cnt.append(val_df[col].isna().sum())
            test_nulls_cnt.append(test_df[col].isna().sum())
            dtypes.append(train_df[col].dtype)
        
        info_df = pd.DataFrame({'Column': columns, 
                                'Train Null Count': train_nulls_cnt, 
                                'Val Null Count': val_nulls_cnt, 
                                'Test Null Count': test_nulls_cnt, 
                                'Dtype': dtypes})

        return info_df

class FeatureTarget:
    def feature_target_split(df: pd.DataFrame, target: str, col_drop=[]):
        y = df[target]
        mask = col_drop + [target]
        if col_drop:
            X = df.drop(columns=mask, axis=1)
        else:
            X = df.drop(columns=target, axis=1)
        return X, y

class Plots:
    def distribution_plots(df:pd.DataFrame, drop: str | list[str]=None, num_features: list[str]=[], cat_features: list[str]=[], bool_features: list[str]=[], time_features: list[str]=[], hue: str=None) -> None:
        # Set the Seaborn style
        sns.set_theme(style="whitegrid")
        columns = df.columns.to_list()
        if drop:
            if type(drop) == str:
                if drop not in columns:
                    raise KeyError(f'{drop} is not in the feature list.')
                else:
                    columns.remove(drop)
            elif type(drop) == list:
                for f in drop:
                    if f not in columns:
                        raise KeyError(f'{f} is not in the feature list.')
                    else:
                        columns.remove(f)

        # Define the plot size and the number of rows and columns in the grid
        num_plots = len(columns)
        rows = (num_plots + 1) // 2  # Calculate the number of rows needed (two plots per row)
        cols = 2  # Two plots per row
        _, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

        # Iterate through the numerical features and create the density plots
        for i, feature_name in enumerate(columns):
            row_idx, col_idx = divmod(i, cols)  # Calculate the current row and column index
            if feature_name in cat_features + bool_features:
                if df[feature_name].nunique() > 20:
                    sns.countplot(df, x=feature_name, ax=axes[row_idx, col_idx], order=df[feature_name].value_counts().iloc[:20].index, hue=hue)
                else:
                    sns.countplot(df, x=feature_name, ax=axes[row_idx, col_idx], order=df[feature_name].unique().sort(), hue=hue)
                axes[row_idx, col_idx].set_title(f'Count Plot of {feature_name}')
                axes[row_idx, col_idx].set_xlabel(feature_name)
                axes[row_idx, col_idx].set_ylabel('Count')
                axes[row_idx, col_idx].bar_label(axes[row_idx, col_idx].containers[0])
            elif feature_name in num_features + time_features:
                sns.histplot(df, x=feature_name, kde=True, ax=axes[row_idx, col_idx], hue=hue, legend=False)
                axes[row_idx, col_idx].set_title(f'Density Plot of {feature_name}')
                axes[row_idx, col_idx].set_xlabel(feature_name)
                axes[row_idx, col_idx].set_ylabel('Density')
            elif feature_name != 'tag':
                raise KeyError(f'{feature_name} is not in the feature list.')
            axes[row_idx, col_idx].tick_params(axis='x', rotation=45)
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots

        plt.show()

class Correlations:
    def features_corr(df: pd.DataFrame, num_features: list) -> None:
        df = df[num_features]
        cmap = sns.color_palette("light:b", as_cmap=True)
        sns.heatmap(df.corr().abs(), cmap=cmap,
                square=True, linewidths=.5, annot=True)
        plt.show()

    def pairwise_corr(data:pd.DataFrame, target: str) -> None:
        df = data.drop(target, axis=1)
        num_features = df.columns[(df.dtypes == 'int64') | (df.dtypes == 'float64')].to_list()
        cat_features = df.columns[df.dtypes == 'object'].to_list()
        features = num_features + cat_features

        # Set the Seaborn style
        sns.set(style="whitegrid")

        # Define the plot size and the number of rows and columns in the grid
        num_plots = len(features) * (len(features) - 1) // 2
        cols = 3  # 3 plots per row
        rows = num_plots // cols + 1  # Calculate the number of rows needed (3 plots per row)
        _, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

        # Iterate through the numerical features and create the density plots
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                row_idx, col_idx = divmod(i * (2 * len(features) - i - 1) // 2 + j - i - 1, cols)  # Calculate the current row and column index
                if features[i] in num_features and features[j] in num_features:
                    sns.scatterplot(data=data, x=features[i], y=features[j], ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Scatter Plot of {features[i]} against {features[j]}')
                    axes[row_idx, col_idx].set_xlabel(features[i])
                    axes[row_idx, col_idx].set_ylabel(features[j])
                elif features[i] in num_features and features[j] in cat_features:
                    sns.countplot(data=data, x=features[j], hue=features[i], ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Count Plot of {features[j]} subject to {features[i]}')
                    axes[row_idx, col_idx].set_xlabel(features[j])
                    axes[row_idx, col_idx].set_ylabel('Count')
                elif features[j] in num_features and features[i] in cat_features:
                    sns.countplot(data=data, x=features[i], hue=features[j], ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Count Plot of {features[i]} subject to {features[j]}')
                    axes[row_idx, col_idx].set_xlabel(features[i])
                    axes[row_idx, col_idx].set_ylabel('Count')
                else:
                    sns.countplot(data=data, x=features[i], hue=features[j], ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Count Plot of {features[i]} subject to {features[j]}')
                    axes[row_idx, col_idx].set_xlabel(features[i])
                    axes[row_idx, col_idx].set_ylabel('Count')
        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plots

        plt.show()
