#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Generate Swiss roll dataset
n_samples = 1000
X, t = make_swiss_roll(n_samples, noise=0.1)

# 2. Plot the Swiss roll dataset
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.Spectral)
ax.set_title("Swiss Roll Dataset")
plt.show()


# In[2]:


from sklearn.decomposition import KernelPCA

# Initialize KernelPCA with different kernels
kpca_linear = KernelPCA(n_components=2, kernel='linear')
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
kpca_sigmoid = KernelPCA(n_components=2, kernel='sigmoid')

# Transform the dataset using each kernel
X_kpca_linear = kpca_linear.fit_transform(X)
X_kpca_rbf = kpca_rbf.fit_transform(X)
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X)


# In[3]:


import matplotlib.pyplot as plt

# Plotting function for comparison
def plot_kpca(X_transformed, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=t, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.xlabel("1st Principal Component")
    plt.ylabel("2nd Principal Component")
    plt.show()

# Plot the results
plot_kpca(X_kpca_linear, "kPCA with Linear Kernel")
plot_kpca(X_kpca_rbf, "kPCA with RBF Kernel")
plot_kpca(X_kpca_sigmoid, "kPCA with Sigmoid Kernel")


# In[5]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Define the pipeline with kPCA and Logistic Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])

# Define the parameter grid for GridSearchCV
param_grid = {
    "kpca__kernel": ["rbf", "sigmoid"],
    "kpca__gamma": np.linspace(0.01, 0.1, 10)
}

# Convert continuous target to discrete labels
t_discrete = np.digitize(t, bins=np.linspace(min(t), max(t), 4)) - 1

# Perform GridSearch with the discrete target
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X, t_discrete)

# Print the best parameters
print("Best parameters found by GridSearchCV:", grid_search.best_params_)


# In[6]:


# Function to plot GridSearchCV results
def plot_grid_search_results(grid_search):
    results = grid_search.cv_results_
    gamma_values = results['param_kpca__gamma'].data
    mean_scores = results['mean_test_score']
    
    plt.figure(figsize=(8, 6))
    plt.plot(gamma_values, mean_scores, marker='o')
    plt.title("Grid Search Results: Accuracy vs Gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Mean Accuracy")
    plt.grid(True)
    plt.show()

# Plot the results of GridSearchCV
plot_grid_search_results(grid_search)


# In[ ]:




