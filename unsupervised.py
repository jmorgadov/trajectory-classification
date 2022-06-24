import feature_vec as fv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from sklearn import metrics

metadata = fv.get_selected_data()
feat_vectors, clss_mask, clss = fv.get_feat_vectors(metadata)

X = StandardScaler().fit_transform(feat_vectors)

count_dict = {i: [0]*5 for i in range(5)}
clss_count = [0]*5

for i in range(len(clss)):
    clss_count[clss[i]] += 1

kmeans = KMeans(n_clusters=5,
                n_init=10,
                init='random',
                tol=1e-4, 
                random_state=170,
                verbose=True)

y_pred = kmeans.fit(X)

for i in range(len(y_pred.labels_)):
    count_dict[y_pred.labels_[i]][clss[i]] += 1
    
for cluster in count_dict.keys():
    for i in range(len(count_dict[cluster])):
        count_dict[cluster][i] /= clss_count[i]
        

# print(count_dict)

#--------------------------------------------------------------------------------
#DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean').fit_predict(X)

print('Number of clusters: {}'.format(len(set(dbscan[np.where(dbscan != -1)]))))
print('Homogeneity: {}'.format(metrics.homogeneity_score(clss, dbscan)))
print('Completeness: {}'.format(metrics.completeness_score(clss, dbscan)))
#print('Mean Silhouette score: {}'.format(metrics.silhouette_score(X, dbscan)))

