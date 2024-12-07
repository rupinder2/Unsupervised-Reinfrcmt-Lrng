import scipy.io as sio

# Load the .mat file
file_path = '/content/umist_cropped.mat'
data = sio.loadmat(file_path)

# Display the keys to understand the structure of the dataset
data.keys()

# Check the structure and size of 'facedat'
facedat = data['facedat']
facedat_shape = facedat.shape

# Check the contents of 'dirnames'
dirnames = data['dirnames']
dirnames_contents = dirnames

facedat_shape, dirnames_contents

# Extract and inspect the first element of 'facedat' to understand its structure
first_face_data = facedat[0, 0]

# Check the type and shape of the first face data instance
type(first_face_data), first_face_data.shape if hasattr(first_face_data, 'shape') else 'No shape attribute'

# Verify the structure and consistency across all entries in 'facedat'
all_shapes = [facedat[0, i].shape for i in range(facedat.shape[1])]

# Check if all shapes are consistent
consistent_shapes = all(shape == all_shapes[0] for shape in all_shapes)

all_shapes, consistent_shapes

# Verify the structure and consistency across all entries in 'facedat'
all_shapes = [facedat[0, i].shape for i in range(facedat.shape[1])]

# Check if all shapes are consistent
consistent_shapes = all(shape == all_shapes[0] for shape in all_shapes)

all_shapes, consistent_shapes

import numpy as np
from sklearn.model_selection import train_test_split

# Normalize pixel values to range [0, 1]
normalized_faces = [facedat[0, i] / 255.0 for i in range(facedat.shape[1])]

# Perform stratified split for training, validation, and test sets
train_data, val_test_data = [], []
val_data, test_data = [], []

# Split for each individual's images
for images in normalized_faces:
    n_images = images.shape[-1]
    indices = np.arange(n_images)

    # Split into training (70%) and remaining (30%)
    train_idx, val_test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    train_data.append(images[..., train_idx])
    val_test_data.append(images[..., val_test_idx])

    # Split remaining into validation (50%) and test (50%)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=42)
    val_data.append(images[..., val_idx])
    test_data.append(images[..., test_idx])

# Combine all data sets into arrays
train_data = np.concatenate(train_data, axis=-1)
val_data = np.concatenate(val_data, axis=-1)
test_data = np.concatenate(test_data, axis=-1)

train_data.shape, val_data.shape, test_data.shape

from sklearn.decomposition import PCA

# Flatten images for PCA
train_flat = train_data.reshape(-1, train_data.shape[-1]).T  # (num_images, pixels)
val_flat = val_data.reshape(-1, val_data.shape[-1]).T
test_flat = test_data.reshape(-1, test_data.shape[-1]).T

# Apply PCA to reduce dimensionality, preserving 95% variance
pca = PCA(0.95, random_state=42)
train_pca = pca.fit_transform(train_flat)
val_pca = pca.transform(val_flat)
test_pca = pca.transform(test_flat)

# Check the reduced dimensionality
train_pca.shape, val_pca.shape, test_pca.shape, pca.n_components_

from sklearn.cluster import KMeans

# Number of clusters (assume one cluster per individual)
n_clusters = len(normalized_faces)

# Apply K-Means clustering on training data
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(train_pca)

# Analyze cluster assignments
unique_clusters, cluster_counts = np.unique(kmeans_labels, return_counts=True)

unique_clusters, cluster_counts

# Analyze the unique labels in training and validation sets
train_labels = kmeans_labels  # Using K-Means as labels for now
val_labels = np.zeros(val_pca.shape[0])  # Placeholder, will need adjustment

# Check consistency across labels in training
unique_train_labels = np.unique(train_labels)

unique_train_labels, len(unique_train_labels)  # Number of unique clusters (should be 20)

# Assign cluster-based labels to validation data using the trained K-Means model
val_labels_corrected = kmeans.predict(val_pca)

# Check the distribution of labels in the validation set
unique_val_labels, val_label_counts = np.unique(val_labels_corrected, return_counts=True)

unique_val_labels, val_label_counts  # Distribution of labels in the validation set

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce PCA-transformed data to 2D for visualization
pca_2d = PCA(n_components=2, random_state=42)
train_2d = pca_2d.fit_transform(train_pca)

# Plot the clusters in 2D space
plt.figure(figsize=(10, 8))
for cluster in np.unique(kmeans_labels):
    cluster_points = train_2d[kmeans_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.6)

plt.title('K-Means Clustering of Training Data (2D PCA Projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid()
plt.show()

import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the neural network architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(train_pca.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(n_clusters, activation='softmax')  # Output layer for 20 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model architecture
model.summary()

history_corrected = model.fit(
    train_pca,
    train_labels,  # Correct training labels (from K-Means clustering)
    validation_data=(val_pca, val_labels_corrected),  # Corrected validation labels
    epochs=50,
    batch_size=16,
    verbose=1
)

# Plot the training and validation accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history_corrected.history['accuracy'], label='Training Accuracy')
plt.plot(history_corrected.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (Corrected Validation Labels)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history_corrected.history['loss'], label='Training Loss')
plt.plot(history_corrected.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (Corrected Validation Labels)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Predict and evaluate on the test set
test_labels_corrected = kmeans.predict(test_pca)  # Correct test labels
test_loss, test_accuracy = model.evaluate(test_pca, test_labels_corrected, verbose=1)

print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

model.save("my_model.keras")

from tensorflow.keras.models import load_model
model = load_model("/content/my_model.keras")

# Predict labels for the test set
test_predictions = model.predict(test_pca)
predicted_labels = np.argmax(test_predictions, axis=1)  # Get the class with the highest probability

# Compare predicted labels with true labels
comparison = list(zip(predicted_labels, test_labels_corrected))

# Display predicted and true labels for the first 10 test samples
for i, (predicted, true) in enumerate(comparison[:10]):
    print(f"Sample {i+1}: Predicted = {predicted}, True = {true}")

test_loss, test_accuracy = model.evaluate(test_pca, test_labels_corrected, verbose=1)
print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")