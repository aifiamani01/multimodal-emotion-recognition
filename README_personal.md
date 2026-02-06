# Personal Notes — Audio Embeddings Step

This file is for **personal understanding** of the first step in the multimodal emotion recognition project.

---

## 1️⃣ Goal of this Step

We are building a **research-ready multimodal emotion recognition project**, and the **first modality we handle is audio**.  

- Dataset: RAVDESS emotional speech audio (actors speaking with different emotions).  
- Objective: Convert raw audio into **features (embeddings)** suitable for neural network models.  
- Why: Neural networks cannot directly understand raw audio; embeddings summarize important patterns like pitch, tone, and emotion cues.

---

## 2️⃣ Why We Use Wav2Vec2

- Wav2Vec2 is a **pretrained model for speech audio**.  
- It has learned to recognize **speech patterns** from thousands of hours of data.  
- Using Wav2Vec2 gives **high-quality audio embeddings** that are better than handcrafted features.  
- This is a standard approach in **research for speech emotion recognition**.

---

## 3️⃣ Resampling Problem

- Wav2Vec2 expects audio at **16 kHz**, but RAVDESS files are at **48 kHz**.  
- Feeding 48 kHz audio would **confuse the model**.  
- Solution: **Resample all audio to 16 kHz** before extracting embeddings.

---

## 4️⃣ What We Did With Each File

For each `.wav` file:

1. Load audio from disk.  
2. Resample to 16 kHz.  
3. Pass through Wav2Vec2 to get a **matrix of embeddings**.  
   - Shape: `(time, features)` — represents audio over time.  
4. Save the embeddings as `.npy` files.  
   - This is faster for future training.  
   - Precomputed embeddings avoid repeatedly processing raw audio.

---

## 5️⃣ Organizing by Actor

- RAVDESS has **24 actors**, each with multiple recordings.  
- Embeddings are saved in folders like:

experiments/audio_embeddings/Actor_01/
experiments/audio_embeddings/Actor_02/
...


- Keeps data **structured and reproducible**, which is crucial in research.

---

## ✅ Summary

- **Raw audio → Resampled → Wav2Vec2 embeddings → Saved as `.npy`**  
- Ready for the **next step**:  
  1. Map embeddings to **emotion labels**.  
  2. Create a **PyTorch dataset/dataloader**.  
  3. Train a baseline **audio emotion classifier**.

