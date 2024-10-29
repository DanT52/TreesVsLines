
#imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor




data = pd.read_csv("AmesHousing.csv")
data = data.drop(columns=['Order', 'PID', 'Neighborhood'])
data = data.fillna(0)



# get dummies for categoricals ...
categorical_features = ['MS Zoning', 'Street', 'Condition 1', 'Roof Matl', 'Heating QC', 'Kitchen Qual', 'Garage Type', 'Paved Drive', 'Sale Condition']
dummified_additional = pd.get_dummies(data[categorical_features], dtype=float)

# create individual variables for each categorical features columns
street = dummified_additional.filter(like='Street').columns.tolist()
condition_1 = dummified_additional.filter(like='Condition 1').columns.tolist()
roof_matl = dummified_additional.filter(like='Roof Matl').columns.tolist()
heating_qc = dummified_additional.filter(like='Heating QC').columns.tolist()
kitchen_qual = dummified_additional.filter(like='Kitchen Qual').columns.tolist()
garage_type = dummified_additional.filter(like='Garage Type').columns.tolist()
paved_drive = dummified_additional.filter(like='Paved Drive').columns.tolist()
sale_condition = dummified_additional.filter(like='Sale Condition').columns.tolist()
zoning = dummified_additional.filter(like='MS Zoning').columns.tolist()


# select all the already numerical columns i plan to use
cols_to_use = ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area']

# concatenate the selected data with dummified_additional and SalePrice
data = pd.concat([dummified_additional, data[cols_to_use], data[['SalePrice']]], axis=1)

# x and y axis data
xs = data.drop( columns = [ 'SalePrice' ] )
ys = data[ 'SalePrice' ]


# column selecter
class SelectColumns( BaseEstimator, TransformerMixin ):
    # pass the function we want to apply to the column 'SalePrice’
    def __init__( self, columns ):
        self.columns = columns
        # don't need to do anything
    def fit( self, xs, ys, **params ):
        return self
    # actually perform the selection
    def transform( self, xs ):
        return xs[ self.columns ]

# for regression model
regressor = TransformedTargetRegressor(
    LinearRegression( n_jobs = -1 ),
    func = np.sqrt,
    inverse_func = np.square
)


#pipeline for linear regression
steps_lr = [
    ('column_select', SelectColumns(['GrLivArea', 'OverallQual'])),
    ('linear_regression', regressor),
]
pipe_lr = Pipeline( steps_lr )




# categoricals = !street, condition_1, !roof_matl, heating_qc, kitchen_qual, garage_type, !paved_drive, sale_condition, zoning
grid_lr = { 
    'column_select__columns': [
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces'],
        
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # [ 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],

        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # [ 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition + zoning
    ],
    'linear_regression': [
        LinearRegression(n_jobs=-1),  # no transformation
        TransformedTargetRegressor(
            LinearRegression(n_jobs=-1),
            func=np.sqrt,
            inverse_func=np.square),
        TransformedTargetRegressor(
            LinearRegression(n_jobs=-1),
            func=np.cbrt,
            inverse_func=lambda y: np.power(y, 3)),
        TransformedTargetRegressor(
            LinearRegression(n_jobs=-1),
            func=np.log,
            inverse_func=np.exp),
    ]
}


# run linear regression model
search_lr = GridSearchCV( pipe_lr, grid_lr, scoring = 'r2', n_jobs = -1, cv = 5)
search_lr.fit( xs, ys )




#decision tree model
pipe_dt = Pipeline([
    ('column_select', SelectColumns(['GrLivArea', 'OverallQual'])),
    ('regression', DecisionTreeRegressor(random_state=5)),
])

grid_dt = {
    'column_select__columns': [
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition + zoning,
        # ['Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces'],
        
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # [ 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],

        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # [ 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition + zoning
    ],
    'regression__max_depth': [
        #None, 
        10, 
        #20, 
        #30
    ],
    'regression__min_samples_split': [
        #2, 
        #5, 
        #10,
        20,
        #30
    ],
    'regression__min_samples_leaf': [
        1, 
        #2, 
        #4
    ]
}

# run decision tree
search_dt = GridSearchCV(pipe_dt, grid_dt, scoring='r2', n_jobs=-1, cv=5)
search_dt.fit(xs, ys)

# random Forest regressor
pipe_rf = Pipeline([
    ('column_select', SelectColumns(['GrLivArea', 'OverallQual'])),
    ('regression', RandomForestRegressor(random_state=5)),
])

grid_rf = {
    'column_select__columns': [
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area']+ condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area']+ condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area']+ condition_1 + heating_qc + kitchen_qual + garage_type + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area']+ condition_1 + heating_qc + kitchen_qual + sale_condition + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area']+ condition_1 + heating_qc + garage_type + sale_condition + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area']+ condition_1 + kitchen_qual + garage_type + sale_condition + zoning,
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area']+ heating_qc + kitchen_qual + garage_type + sale_condition + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces'],
        
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # [ 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],

        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # [ 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition + zoning
    ],
    'regression__n_estimators': [
        #100,
        #200,
        300,
        #400
    ],
    'regression__max_depth': [
        #10, 
        #20, 
        30,
        #40
    ],
    #'regression__max_features': ['auto', 'sqrt']
}

# run random forest 
search_rf = GridSearchCV(pipe_rf, grid_rf, scoring='r2', n_jobs=-1, cv=5)
search_rf.fit(xs, ys)



# gradient boosting regressor
pipe_gb = Pipeline([
    ('column_select', SelectColumns(['GrLivArea', 'OverallQual'])),
    ('regression', GradientBoostingRegressor(random_state=5)),
])

grid_gb = {
    'column_select__columns': [
        #['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + garage_type + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + sale_condition + zoning,
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + garage_type + sale_condition + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + kitchen_qual + garage_type + sale_condition + zoning,
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + heating_qc + kitchen_qual + garage_type + sale_condition + zoning,
        # ['Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Remod/Add', 'Fireplaces'],
        
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # [ 'Overall Qual', 'Year Built', 'Garage Area', 'Full Bath', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],

        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # [ 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'],
        # ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Area', 'Year Remod/Add', 'Fireplaces', 'Lot Area'] + condition_1 + heating_qc + kitchen_qual + garage_type + sale_condition + zoning
    ],
    'regression__n_estimators': [
        #100, 
        #200, 
        300,
        #400
    ],
    'regression__learning_rate': [
        #0.01, 
        0.1,
        #0.15, 
        #0.2
    ],
    'regression__max_depth': [
        #2, 
        3, 
        #5, 
        #7
    ]
}

# run gradient boosting regressor
search_gb = GridSearchCV(pipe_gb, grid_gb, scoring='r2', n_jobs=-1, cv=5)
search_gb.fit(xs, ys)


# print all outputs

# Linear regression
print("Linear regression:")
print(f"R-squared: {search_lr.best_score_}")
print(f"Best params: {search_lr.best_params_}")
print("\n")

# Random forest
print("Random forest:")
print(f"R-squared: {search_rf.best_score_}")
print(f"Best params: {search_rf.best_params_}")
print("\n")

# Decision tree
print("Decision tree:")
print(f"R-squared: {search_dt.best_score_}")
print(f"Best params: {search_dt.best_params_}")
print("\n")

# Gradient boosting
print("Gradient boosting:")
print(f"R-squared: {search_gb.best_score_}")
print(f"Best params: {search_gb.best_params_}")


