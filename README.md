# Trees vs Lines

This project compares the performance of different regression models on the Ames Housing dataset. The models used include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Data Preparation

The dataset is preprocessed by:
1. Dropping unnecessary columns.
2. Filling missing values with zero.
3. Encoding categorical features using one-hot encoding.

## Models and Pipelines

Each model is wrapped in a scikit-learn `Pipeline` and hyperparameters are tuned using `GridSearchCV`. The target variable `SalePrice` is transformed using various functions for linear regression.

## Results

The performance of each model is evaluated using the R-squared metric. The best hyperparameters for each model are also identified.

## Usage

To run the models and see the results, execute the script `treesVsLines.py`.

```bash
python treesVsLines.py
```

## Dependencies

- pandas
- numpy
- scikit-learn

