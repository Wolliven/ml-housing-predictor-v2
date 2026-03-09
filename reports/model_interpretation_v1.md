# Model Interpretation v1

## Coefficients

Feature coefficients (sorted by importance):

* Latitude: -0.817979
* MedInc: 0.797270
* Longitude: -0.773951
* households: 0.497026
* Population: -0.450589
* bedrooms_per_room: 0.208294
* AveRooms: 0.162728
* HouseAge: 0.149142
* AveOccup: 0.119120
* people_per_bedroom: -0.117046
* AveBedrms: -0.079087

## Observations

* Geographic features (**Latitude** and **Longitude**) have the strongest influence on predictions, indicating that location captures a large portion of house price variation across California.
* **Median income (MedInc)** is the strongest positive predictor, which aligns with expectations since wealthier districts typically have higher housing prices.
* The engineered feature **households** appears among the most influential predictors, suggesting that the estimated number of households in a district carries meaningful information about housing value.
* The engineered ratio **people_per_bedroom** has a negative coefficient, indicating that higher household crowding tends to correlate with lower house prices.
* **bedrooms_per_room** has a positive coefficient, suggesting that housing composition (relative bedroom count) contributes to the model's prediction signal.
* **Population** has a moderately strong negative coefficient, which may indicate that densely populated districts tend to have lower average housing values in this dataset.
* Other features such as **AveRooms**, **HouseAge**, and **AveOccup** have smaller but still positive contributions to predicted house value.

## Summary

* The model relies most strongly on **location and income**, which dominate the prediction signal.
* Engineered features related to **household structure and crowding** also contribute meaningful information to the model.
* The coefficient analysis supports the results observed in the feature engineering experiment, where the additional features improved Ridge regression performance.
