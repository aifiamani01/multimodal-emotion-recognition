import os
import numpy as np

# Paths
embeddings_root = "experiments/audio_embeddings"  # where your .npy embeddings are
output_path = "data/processed"                     # where we'll save X and y

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)


# RAVDESS emotion mapping
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}



X_list = []
y_list = []

# Loop over each actor folder
actors = sorted(os.listdir(embeddings_root))  # Actor_01 ... Actor_24

for actor in actors:
    actor_path = os.path.join(embeddings_root, actor)
    embedding_files = [f for f in os.listdir(actor_path) if f.endswith(".npy")]

    for emb_file in embedding_files:
        # Load embedding
        emb_path = os.path.join(actor_path, emb_file)
        embedding = np.load(emb_path)
        X_list.append(embedding)

        # Extract emotion from filename
        # e.g., 03-01-06-01-02-01-12.npy
        emotion_id = emb_file.split("-")[2]
        y_list.append(emotion_map[emotion_id])


# Convert lists to numpy arrays
X = np.array(X_list, dtype=object)  # object because embeddings have variable lengths
y = np.array(y_list)

# Save
np.save(os.path.join(output_path, "X_embeddings.npy"), X)
np.save(os.path.join(output_path, "y_labels.npy"), y)

print("✅ Dataset built successfully!")
print(f"X shape: {X.shape}, y shape: {y.shape}")
