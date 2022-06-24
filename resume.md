# Artificial intelligence for trajectory classification with and without features

The principal goal is to analyze clasification tecniques for trajectories calculating
features and using the raw data. This way, we can see the importance of using
the estimated features.

## Using features

### Features used

- Distance
- Velocity
- Acceleration
- Stop rate
- Angle
- Turning angle
- Heading change rate

As some of this features are vectors which can be of different size, we
estimated some observables for them such as: mean value, max value, min value,
standard deviation, interquartile ranges, etc.

The final vector of features for each trajectory contains 51 elements.

Below the results for each model is shown:

| Algorithm									| Accuracy	|
| :--										|	   --:	|
| KNN										|    70.54%	|
| SVM (Polinomial Kernel)					|    71.14% |
| SVM (Sigmoid Kernel)						|    75.22% |
| SVM (RBF Kernel)							|    83.68% |
| Decision Tree							    |	 83.08% |
| **Random Forest**							|**90.04%** |
| Random Forest (Standar Scaler)  			|    89.55% |
| Random Forest (PCA with 15 components)  	|    80.39% |


## Using raw data

