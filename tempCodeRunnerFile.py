import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#1 load the dataset
housing = pd.read_csv("housing.csv")


#2 create a stratified test set
housing["income_cat"] = pd.cut(housing["median_income"]
                               ,bins=[0.0,1.5,3,4.5,6,
                                      np.inf],labels=[1,2,3,4,5]);

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing["income_cat"]):
    strate_train_set = housing.loc[train_index].drop("income_cat",axis=1)
    strate_test_set = housing.loc[test_index].drop("income_cat",axis=1)

# we will work on the copy of training dataset
housing = strate_train_set.copy()

# seprate Features and labels
housing_label = housing["median_house_value"]
housing_features = housing.drop("median_house_value",axis=1)


#4 seprate numerical and categorical columns
num_attribs = housing_features.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


#5 now let's meke the pipeline

#  for numerical columns
num_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())#standardscaler robut to outliers
])

#  for categorical columns
cat_pipeline = Pipeline([
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

#construct the full pipeline
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",cat_pipeline,cat_attribs)
])

#6 Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
#print(housing_prepared.shape)


#7 Train the model

#linear_regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_label)
lin_preds = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_label,lin_preds)
lin_rmse = -cross_val_score(lin_reg,housing_prepared,housing_label,scoring="neg_root_mean_squared_error",cv=10)
#print(f"The root mean square error for linear regeression is {lin_rmse}")

print(pd.Series(lin_rmse).describe())
#print(f"The root mean square error for linearregression is {lin_rmse}")


#decision tree_regressor model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_label)
dec_preds = dec_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_label,dec_preds)
dec_rmse = -cross_val_score(dec_reg,housing_prepared,housing_label,scoring="neg_root_mean_squared_error",cv=10)
#print(f"The root mean square error for decision tree is {dec_rmse}")

print(pd.Series(dec_rmse).describe())


#random forest model
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_label)
forest_preds = forest_reg.predict(housing_prepared)
#forest_rmse = root_mean_squared_error(housing_label,forest_preds)
random_forest_rmse = -cross_val_score(forest_reg,housing_prepared,housing_label,scoring="neg_root_mean_squared_error",cv=10)
#print(f"The root mean square error for random forest  is {forest_rmse}")


#evaluate the model
print(pd.Series(random_forest_rmse).describe())
#print(f"The root mean square error for random forest is {forest_rmse}")