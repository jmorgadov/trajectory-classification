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

### Results

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

An LSTM model was created for the classification. The model structure is:

| Layers |
| :-- | 
| LSTM (128 units with return sequence) |
| LSTM (64 units with return sequence) |
| Bidirectional LSTM (32 units) | 
| Dense (15 units, relu activation) |
| Dense (5 or 3 units, softmax activation) |

### Results

| Classes						| Test Accuracy |
| :--							| :--			|
| bike, car, bus, train, walk	| 47.16%		|
| **bike, car, walk**			| **60.60%**	|

## Final observations

As we can see the supervised algorithms using the features had the best results,
being 90.04% the max accuracy achieved.

## Recommendations

It is recommended for future investigations to analyze the LSTM model in depth and
also try the classification using a transformer.