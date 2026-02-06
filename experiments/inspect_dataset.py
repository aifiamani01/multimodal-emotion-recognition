import numpy as np
import matplotlib.pyplot as plt

# Paths to dataset
X_path = "data/processed/X_embeddings.npy"
y_path = "data/processed/y_labels.npy"

# Load
X = np.load(X_path, allow_pickle=True)  # embeddings were object dtype
y = np.load(y_path)

print(f"✅ Loaded X: {X.shape}, y: {y.shape}")
print("Sample labels:", y[:10])


# Pick the first sample
sample_embedding = X[0]
sample_label = y[0]

print(f"Sample label: {sample_label}")
print(f"Embedding shape: {sample_embedding.shape}")

# Plot the embedding (time x features)
plt.figure(figsize=(12, 4))
plt.imshow(sample_embedding.T, aspect="auto", origin="lower")
plt.colorbar()
plt.title(f"Embedding for emotion: {sample_label}")
plt.xlabel("Time steps")
plt.ylabel("Feature dimension")
plt.show()


# Count samples per emotion
from collections import Counter
label_counts = Counter(y)
print("Number of samples per emotion:")
for emotion, count in label_counts.items():
    print(f"{emotion}: {count}")
