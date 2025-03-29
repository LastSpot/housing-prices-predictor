# %%
import numpy as np
import pandas as pd
import math
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from model import MLPNeuralNetwork
# %%
def nomalization(data: pd.DataFrame, label_col: str):
    # Make a copy to avoid modifying the original
    data_to_normalize = data.copy()
    
    # Only drop the label column if it exists and is specified
    if label_col and label_col in data.columns:
        data_to_normalize = data.drop(columns=[label_col])
        
    features_max = data_to_normalize.max()
    features_min = data_to_normalize.min()
    
    # Handle division by zero (when max == min)
    range_values = features_max - features_min
    range_values = range_values.replace(0, 1)  # Replace zeros with 1 to avoid division by zero
    
    normalized_data = 2 * (data_to_normalize - features_min) / range_values - 1
    
    # Add back the label column if it exists
    if label_col and label_col in data.columns:
        normalized_data[label_col] = data[label_col]
        
    return normalized_data

# %%
def preprocess_data(data: pd.DataFrame, columns_dtype: dict, label_col: str, fit_encoders=False):
    imputer = KNNImputer(n_neighbors=int(np.sqrt(len(data))))
    for col in data.columns:
        if col in columns_dtype and columns_dtype[col] == 1: # categorical
            data[col] = data[col].fillna('unknown')
        elif col in columns_dtype and columns_dtype[col] == 0: # numerical
            imputed_values = imputer.fit_transform(data[[col]])
            data[col] = imputed_values.flatten()
                
    categorical_cols = [col for col in data.columns if col in columns_dtype and columns_dtype[col] == 1]
    numerical_cols = [col for col in data.columns if col in columns_dtype and columns_dtype[col] == 0]
    
    # For train data, fit and transform. For test data, just transform
    if fit_encoders:
        global encoder
        encoded_categorical_data = encoder.fit_transform(data[categorical_cols])
    else:
        encoded_categorical_data = encoder.transform(data[categorical_cols])
        
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    
    for feature in encoded_feature_names:
        if feature not in columns_dtype:
            columns_dtype[feature] = 1
    
    categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoded_feature_names, index=data.index)
    
    normalized_numerical_data = nomalization(data[numerical_cols], label_col)
    
    df = pd.concat([categorical_df, normalized_numerical_data], axis=1)

    return df

# %%
def df_to_np(df: pd.DataFrame, label_col: str):
    if label_col:
        features_data = df.drop(columns=[label_col]).to_numpy()
        label_data = df[label_col].to_numpy()
        return features_data, label_data
    else:
        return df.to_numpy()

# %%
def k_fold(features: np.ndarray, labels: np.ndarray, k: int):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    return kf.split(features, labels)
# %%
train_df = pd.read_csv('home-data-for-ml-course/train.csv')
test_df = pd.read_csv('home-data-for-ml-course/test.csv')

# %%
# Print basic information about the training and test datasets
print("Training dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)

# Print the number of columns in each dataset
print(f"Number of columns in training dataset: {train_df.shape[1]}")
print(f"Number of columns in test dataset: {test_df.shape[1]}")


# Check for missing values
print("\nMissing values in training dataset:")
print(train_df.isnull().sum().sort_values(ascending=False).head(10))

print("\nMissing values in test dataset:")
print(test_df.isnull().sum().sort_values(ascending=False).head(10))

# %%
# %%
# 0 is numerical, 1 is categorical
columns_dtypes = {
    'MSSubClass': 1,
    'MSZoning': 1,
    'LotFrontage': 0,
    'LotArea': 0,
    'Street': 1,
    'Alley': 1,
    'LotShape': 1,
    'LandContour': 1,
    'Utilities': 1,
    'LotConfig': 1,
    'LandSlope': 1,
    'Neighborhood': 1,
    'Condition1': 1,
    'Condition2': 1,
    'BldgType': 1,
    'HouseStyle': 1,
    'OverallQual': 1,
    'OverallCond': 1,
    'YearBuilt': 0,
    'YearRemodAdd': 0,
    'RoofStyle': 1,
    'RoofMatl': 1,
    'Exterior1st': 1,
    'Exterior2nd': 1,
    'MasVnrType': 1,
    'MasVnrArea': 0,
    'ExterQual': 1,
    'ExterCond': 1,
    'Foundation': 1,
    'BsmtQual': 1,
    'BsmtCond': 1,
    'BsmtExposure': 1,
    'BsmtFinType1': 1,
    'BsmtFinSF1': 0,
    'BsmtFinType2': 1,
    'BsmtFinSF2': 0,
    'BsmtUnfSF': 0,
    'TotalBsmtSF': 0,
    'Heating': 1,
    'HeatingQC': 1,
    'CentralAir': 1,
    'Electrical': 1,
    '1stFlrSF': 0,
    '2ndFlrSF': 0,
    'LowQualFinSF': 0,
    'GrLivArea': 0,
    'BsmtFullBath': 0,
    'BsmtHalfBath': 0,
    'FullBath': 0,
    'HalfBath': 0,
    'BedroomAbvGr': 0,
    'KitchenAbvGr': 0,
    'KitchenQual': 1,
    'TotRmsAbvGrd': 0,
    'Functional': 1,
    'Fireplaces': 0,
    'FireplaceQu': 1,
    'GarageType': 1,
    'GarageYrBlt': 0,
    'GarageFinish': 1,
    'GarageCars': 0,
    'GarageArea': 0,
    'GarageQual': 1,
    'GarageCond': 1,
    'PavedDrive': 1,
    'WoodDeckSF': 0,
    'OpenPorchSF': 0,
    'EnclosedPorch': 0,
    '3SsnPorch': 0,
    'ScreenPorch': 0,
    'PoolArea': 0,
    'PoolQC': 1,
    'Fence': 1,
    'MiscFeature': 1,
    'MiscVal': 0,
    'MoSold': 0,
    'YrSold': 0,
    'SaleType': 1,
    'SaleCondition': 1,
    'SalePrice': 0
}

# %%
# Store encoders to ensure consistency between train and test processing
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
# Store min/max values for consistent normalization
numerical_min_max = {}

# %%
preprocess_transformer = FunctionTransformer(func=preprocess_data, kw_args={'columns_dtype': columns_dtypes, 'label_col': 'SalePrice', 'fit_encoders': True})
np_transformer = FunctionTransformer(func=df_to_np, kw_args={'label_col': 'SalePrice'})

# %%
pipeline = Pipeline(steps=[
    ('preprocess', preprocess_transformer),
    ('df_to_np', np_transformer)
])

train_features, train_labels = pipeline.fit_transform(train_df.drop(columns=['Id']))

print('Data preprocessed')
print('Dataframe turned into numpy arrays')
print('Training features shape: ', train_features.shape)
print('Training labels shape: ', train_labels.shape)

# %%
test_preprocess_transformer = FunctionTransformer(func=preprocess_data, kw_args={'columns_dtype': columns_dtypes, 'label_col': None, 'fit_encoders': False})
test_np_transformer = FunctionTransformer(func=df_to_np, kw_args={'label_col': None})

# %%
test_data_pipeline = Pipeline(steps=[
    ('preprocess', test_preprocess_transformer),
    ('df_to_np', test_np_transformer)
])

test_features = test_data_pipeline.fit_transform(test_df.drop(columns=['Id']))
test_id = df_to_np(test_df['Id'], None)

print('Testing data preprocessed')
print('Dataframe turned into numpy arrays')
print('Testing features shape: ', test_features.shape)
print('Testing ids shape: ', test_id.shape)

# %%
k_fold_index = []
for train_index, test_index in k_fold(train_features, train_labels, k=7):
    k_fold_index.append([train_index, test_index])

params = {
    'alphas': [0.6, 0.7, 0.8, 0.9],
    'lambdas': [0, 0.1, 0.2, 0.3],
    'epsilons': [math.pow(math.e, -7), math.pow(math.e, -6), math.pow(math.e, -5)],
    'hidden_sizes': [5, 10],
    'neurons_per_layer': [10, 20],
    'early_stopping_threshold': 150000,
    'early_stopping_folds': 4
}

# %%
neural_network = MLPNeuralNetwork(len(train_features[0]), 1)
neural_network.get_info()

# %%
neural_network.grid_search(train_features, train_labels, params, k_fold_index)

# %%
neural_network.fit(train_features, train_labels)

# %%
housing_price_prediction = neural_network.predict(test_id,test_features)

# %%
print(housing_price_prediction.shape)

# %%
filename = 'submission.csv'
housing_price_prediction.to_csv(filename, index=False)