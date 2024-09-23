#!/usr/bin/env python
# coding: utf-8

# In[22]:


import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

# 1) retrieve and load the dataset
with open(r'C:\Users\isrup\Downloads\mnist_784.arff') as f:
    dataset = arff.load(f)

# Extract data and labels
data = np.array(dataset['data'])
X, y = data[:, :-1], data[:, -1].astype(int)  # Last column is the target labels

# Check data shape
print(X.shape, y.shape)


# In[25]:


# Convert X to float
X = X.astype(float)

# Display each digit
for i in range(20):
    plt.imshow(X[i].reshape(28, 28), cmap='binary')
    plt.title(f"Digit: {y[i]}")
    plt.show()


# In[26]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# output the explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)


# In[27]:


#plot principle components onto a 1d hyperplane
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='plasma', s=1)
plt.colorbar()
plt.title('Projections of 1st and 2nd Principal Components')
plt.show()


# In[28]:


# using incremental PCA to reduce the dimensionality 
ipca = IncrementalPCA(n_components=154, batch_size=200)
X_ipca = ipca.fit_transform(X)


# In[29]:


# Inverse transform the compressed data
X_ipca_inverse = ipca.inverse_transform(X_ipca)

# Display original and compressed digits side by side
for i in range(5):
    plt.subplot(1, 2, 1)
    plt.imshow(X[i].reshape(28, 28), cmap='binary')
    plt.title("Original")
    
    plt.subplot(1, 2, 2)
    plt.imshow(X_ipca_inverse[i].reshape(28, 28), cmap='binary')
    plt.title("Compressed")
    plt.show()


# In[ ]:




