# Feature Engineering Experiment v1

## New Features
- households = Population / AveOccup
- people_per_bedroom = AveOccup / AveBedrms
- bedrooms_per_room = AveBedrms / AveRooms

## Metrics

### Linear
R2: 0.5699
MAE: 0.5176
RMSE: 0.7568

### Ridge
R2: 0.6085
MAE: 0.5171
RMSE: 0.7220

## Error stats

### Linear
mean error: 0.004367871319974438
standard deviation: 0.7567669934756523
max error: 5.807098114946595
min error: -38.813015669095954

### Ridge
mean error: mean: 0.00399345819701428
standard deviation: 0.7220301917627373
max error: 5.81734909479
min error: -20.896026184801954

## Comparison vs Baseline

- Feature engineering improved Ridge regression significantly, increasing R² from 0.58 to 0.61 and reducing both RMSE and MAE.
- Linear regression performance slightly degraded in terms of R² and RMSE, although MAE improved slightly.
- The engineered features appear to introduce correlations between variables, which Ridge handles better due to regularization.

## Observations

- Ridge regression clearly benefits from the engineered features, while Linear regression becomes slightly less stable.
- Mean error remains close to zero in both models, indicating no significant prediction bias.
- Some extreme prediction errors increased significantly due to ratio features producing very large values when denominators are small.
- The worst predictions correspond to districts with extremely high `AveOccup` values, leading to very large `people_per_bedroom` ratios.
- This suggests that ratio-based features may require additional safeguards or normalization to prevent unstable values.
