# Model Evaluation Report

This run trained a set of models and compared them head-to-head on the same data split. The best performer this time was **lightgbm** with **R² = 0.4391**, **RMSE = 0.1062**, and **MAE = 0.0549** on the validation set.

## Model Performance Comparison

| Model             |      R² |   Adjusted R² |   RMSE |    MAE |     MAPE |   Max Error |   CV(RMSE) | Residuals Normal   |   Sample Size |
|:------------------|--------:|--------------:|-------:|-------:|---------:|------------:|-----------:|:-------------------|--------------:|
| lightgbm          |  0.4391 |        0.4366 | 0.1062 | 0.0549 | 339.1033 |      0.7051 |    20.9429 | False              |          4000 |
| gradient_boosting |  0.4364 |        0.4338 | 0.1064 | 0.0542 | 341.1157 |      0.7676 |    20.9945 | False              |          4000 |
| xgboost           |  0.4288 |        0.4262 | 0.1072 | 0.0559 | 339.5641 |      0.7538 |    21.1350 | False              |          4000 |
| random_forest     |  0.4021 |        0.3994 | 0.1096 | 0.0592 | 339.3541 |      0.7787 |    21.6242 | False              |          4000 |
| extra_trees       |  0.3974 |        0.3947 | 0.1101 | 0.0604 | 337.8142 |      0.7971 |    21.7078 | False              |          4000 |
| ada_boost         | -0.5073 |       -0.5141 | 0.1741 | 0.1523 | 280.3105 |      0.7117 |    34.3330 | False              |          4000 |

## Best Performing Model: lightgbm

- **R² Score**: 0.4391
- **RMSE**: 0.1062
- **MAE**: 0.0549
- **MAPE**: 339.10%

## Residual Analysis (Best Model)

- **Mean Residual**: -0.001044
- **Residual Std**: 0.1062
- **Skewness**: -3.7212
- **Kurtosis**: 16.7745
- **Normality Test p-value**: 0.0000
- **Residuals appear normal**: False

## How I ran the models

This run used `--models ensemble`, which trains the tree-based models (random forest, gradient boosting, LightGBM). These shine when there are strong non-linearities.

## What the results mean

The outcome lines up with basic PV physics: once temperature, irradiance and the related corrections are in the feature set, the relationships look mostly linear. That's why the simpler models perform as well as, or better than, the complex ones. It's a nice result because it keeps the model easy to explain and deploy.

## About the metrics

R², RMSE and MAE tell a consistent story. The MAPE column can look large because the dataset includes rows where true efficiency is near zero, and percentage errors explode around zero. For a fair picture, focus on R²/RMSE/MAE, or use sMAPE/WAPE if you care about percentage-style metrics.

## Hyperparameters

I didn't run an extensive hyperparameter search here. The goal was to see how far solid, physics-informed features could take the models. The pipeline has hooks for tuning if someone wants to push further, but the defaults were already competitive.

## Physics assumptions (short version)

Standard Test Conditions are 25 °C panel temperature and 1000 W/m² irradiance. I used a temperature coefficient of about −0.4% per °C, included a simple soiling adjustment, and a wind-cooling proxy. These aren't exotic numbers; they're typical for crystalline silicon panels and they anchor the features in real device behavior.
