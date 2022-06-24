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

| Model 									| Accuracy	|
| :--										|	   --:	|
| KNN										|    70.54%	|
| SVM (Polinomial Kernel)					|    71.14% |
| SVM (Sigmoid Kernel)						|    75.22% |
| SVM (RBF Kernel)							|    83.68% |
| Decision Tree							    |	 83.08% |
| **Random Forest**							|**90.04%** |
| Random Forest (Standard Scaler)  			|    89.55% |
| Random Forest (PCA with 15 components)  	|    80.39% |

#### Unsupervised algorithms

We tried unsupervised models like: K-means, DBSCAN and OPTICS. For this models
we calculated how many different classes where in each cluster and some metrics.

We tested the models with different setups (different parameters) but they
didn't show good results. The best clusterization was from the OPTICS (taking
only the classes car, walk and bike) model but the metrics showed that the data
has too many outliers (few trajectories per cluster) and all the clusters where
too close (negative silhouette -0.45).

## Using raw data

An LSTM model was created for the classification. The model structure is:

| Layers |
| :-- | 
| LSTM (128 units with return sequence) |
| LSTM (64 units with return sequence) |
| Bidirectional LSTM (32 units) | 
| Dense (15 units, relu activation) |
| Dense (5 or 3 units, softmax activation) |

We notice that the classes: car, bus and train where the most confusing ones to
classify for the algorithm, so we leaved only the class car of the aforementioned.
(Thta's why the final Dense layer has 5 of 3 units).

### Results

| Classes						| Test Accuracy |
| :--							| :--			|
| bike, car, bus, train, walk	| 47.16%		|
| **bike, car, walk**			| **60.60%**	|

## Conclusions

As we can see the supervised algorithms using the features had the best results,
being **90.04%** the max accuracy achieved.

## Recommendations

It is recommended for future investigations to analyze the LSTM model in depth and
also try the classification using a transformer.
