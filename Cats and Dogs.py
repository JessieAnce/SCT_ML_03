import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Path to extracted train images from Kaggle dataset
DATA_DIR = "path_to_train_folder"  # e.g., "./dogs-vs-cats/train"
IMG_SIZE = 64

# Step 1: Load and preprocess images
def load_images(data_dir, limit=2000):  # Load a limited number for speed
    data = []
    labels = []
    files = os.listdir(data_dir)
    
    for file in tqdm(files[:limit]):
        label = 1 if "dog" in file else 0
        path = os.path.join(data_dir, file)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0  # Normalize
        data.append(img.flatten())  # Flatten to 1D
        labels.append(label)
    
    return np.array(data), np.array(labels)

print("Loading images...")
X, y = load_images(DATA_DIR, limit=2000)

# Step 2: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train SVM classifier
print("Training SVM...")
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Step 4: Evaluate model
y_pred = svm.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
