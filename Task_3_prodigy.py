!pip install numpy pandas matplotlib scikit-learn opencv-python


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Set the path to your dataset
dataset_path = r'Path to Your dataset folder it should looks like "C:\Users\User name\Downloads\PetImages" '

# Initialize lists to store images and labels
images = []
labels = []

# Define image size
img_size = 64


# Load images
for label in ['cat', 'dog']:
    folder_path = os.path.join(dataset_path, label)
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (img_size, img_size))
            images.append(image)
            labels.append(0 if label == 'cat' else 1)


# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the images
images = images / 255.0


# Display some images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])
    plt.xlabel('Cat' if labels[i] == 0 else 'Dog')
plt.show()


# Flatten the images
n_samples, img_size, _, n_channels = images.shape
images_flat = images.reshape((n_samples, img_size * img_size * n_channels))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)


# Initialize the SVM model
svm = SVC(kernel='linear')

# Train the model
svm.fit(X_train, y_train)


# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(report)


