#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np

# 1) load the dataset
olivetti_faces = fetch_olivetti_faces()
X = olivetti_faces.data  # Feature matrix
y = olivetti_faces.target  # Labels (person ID)

# 2) Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 3) Use k-fold cross-validation to train an SVM classifier to predict the person in each picture
kf = StratifiedKFold(n_splits=5)
svm_model = SVC(kernel='linear')

# Perform cross-validation training
scores = []
for train_index, val_index in kf.split(X_train, y_train):
    X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    svm_model.fit(X_fold_train, y_fold_train)
    scores.append(svm_model.score(X_fold_val, y_fold_val))

# Average cross-validation score
avg_cv_score = np.mean(scores)
print(f'Average cross-validation score: {avg_cv_score}')


# 4) Use K-Means to reduce dimensionality with PCA, choose number of clusters using silhouette score
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_train)
kmeans = KMeans(n_clusters=40, random_state=42)
kmeans.fit(X_pca)
cluster_labels = kmeans.labels_

# Calculate silhouette score to choose optimal number of clusters
silhouette_avg = silhouette_score(X_pca, cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')


# 5) Train classifier using the reduced set
svm_model_pca = SVC(kernel='linear')
kf_pca = StratifiedKFold(n_splits=5)
scores_pca = []
for train_index, val_index in kf_pca.split(X_pca, y_train):
    X_fold_train_pca, X_fold_val_pca = X_pca[train_index], X_pca[val_index]
    y_fold_train_pca, y_fold_val_pca = y_train[train_index], y_train[val_index]
    svm_model_pca.fit(X_fold_train_pca, y_fold_train_pca)
    scores_pca.append(svm_model_pca.score(X_fold_val_pca, y_fold_val_pca))

avg_cv_score_pca = np.mean(scores_pca)
print(f'Average cross-validation score (PCA): {avg_cv_score_pca}')

# 6) Apply DBSCAN for clustering
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan_labels = dbscan.fit_predict(X)


# Check the number of unique clusters
unique_clusters = len(np.unique(dbscan_labels))
print(f'Number of unique clusters: {unique_clusters}')


# In[ ]:




