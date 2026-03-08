# Baseline Analysis (v2.0.0)

## Metrics

### Linear
R2: 0.5807
MAE: 0.5475
RMSE: 0.7472

### Ridge
R2: 0.5802
MAE: 0.5477
RMSE: 0.7476

## Error stats

### Linear
mean error: 0.004767687639465916
standard deviation: 0.7471910666455259
max error: max: 11.031102630328052
min error: min: -7.260453292958463

### Ridge
mean error: mean: 0.00352093411730337
standard deviation: 0.7476523100475282
max error: max: 10.267199773964144
min error: min: -7.58562065045863



## Observations
- Linear Regression and Ridge Regression performed almost identically, so regularization does not seem to provide a meaningful benefit with the current feature set.
- The model explains about 58% of the variance in house prices, which suggests it captures useful signal but still leaves a large part of the variation unexplained.
- The average absolute error is about 0.55 in MedHouseVal units, meaning the model is off by roughly $55,000 on average.
- Mean error is very close to zero, so there is no strong overall overprediction or underprediction bias.
- Some extreme prediction errors are much larger than the average error, which suggests the model struggles with outliers or unusual districts.
- Several of the worst-predicted rows contain very unusual feature values, especially extreme averages such as very high AveRooms or AveOccup.
- These results suggest that the next improvement step should focus on feature engineering and deeper error analysis rather than changing from Linear Regression to Ridge.