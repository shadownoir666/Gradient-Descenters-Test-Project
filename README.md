# 🎙️ Adaptive Speaker Diarization System

An end-to-end **speaker diarization pipeline** built for real-world audio — adaptive, explainable, and fully unsupervised.

This project was built during **CodeSprint (Robomania)**, where it secured **2nd place** 🥈.  
The focus was on engineering a robust system rather than a black-box demo.

---

## 🚀 Demo




https://github.com/user-attachments/assets/b2d26255-eac2-45d3-9284-df69c45ebfac




---

## 🧠 Problem Statement

Given an audio file:
- Identify **who spoke**
- **When** they spoke
- **What** they said
- **How** they sounded (emotion)

All **without labeled data** and **without knowing the number of speakers beforehand**.

Speaker diarization is especially hard due to:
- Speaker overlaps
- Short interruptions
- Similar voice profiles
- Noisy or real-world audio

---

## 🧩 Solution Overview

We built a **modular diarization pipeline** that combines:
- Speech detection
- Overlap-aware segmentation
- Speaker embeddings
- Adaptive clustering
- Post-processing speaker merging
- Emotion analysis per speaker

The system automatically adapts to different audio conditions and conversation styles.

---

## 🛠️ Tech Stack

### Backend / ML
- **Silero VAD** – Speech Activity Detection  
- **Whisper Transformer** – Transcription & diarization support  
- **Resemblyzer** – Speaker embeddings  
- **Speech Emotion Recognition (SER)** – Emotion analysis per segment  
- **Adaptive Clustering (custom-built)**  
- Python, PyTorch, NumPy, scikit-learn  

### Frontend
- **Streamlit** – Lightweight interactive UI  

---

## 🔧 System Architecture

### 1️⃣ Speech Detection & Audio Preprocessing
- Speech regions detected using **Silero VAD**
- Audio standardized via:
  - Mono conversion
  - Fixed sample-rate resampling
  - Normalization and segmentation

This ensures stable embeddings and reduces downstream noise.

---

### 2️⃣ Overlap-Based Segmentation Strategy
Instead of naïve hard cuts, we used **overlapping audio windows**:
- Preserves speaker context at segment boundaries
- Reduces speaker fragmentation
- Improves diarization stability during fast speaker switches

This significantly improved results compared to standard chunking.

---

### 3️⃣ Speaker Embeddings
- Each speech segment is converted into a **high-dimensional voice embedding** using **Resemblyzer**
- Embeddings are L2-normalized
- Cosine distance is used for similarity comparisons

These embeddings form the foundation of clustering.

---

## 🧠 Adaptive Clustering (Core Idea)

Since we had:
- ❌ No labeled dataset
- ❌ No fixed number of speakers

We designed an **adaptive clustering algorithm** instead of using a static threshold.

### How it works:
- Compute **pairwise distance statistics** (quartiles, median) from embeddings
- Generate multiple **candidate distance thresholds**
- Apply **Agglomerative Clustering** with:
  - Cosine distance
  - Average linkage
  - `n_clusters=None` (speaker count emerges naturally)

### Evaluation & Selection:
Each clustering result is scored using:
- **Silhouette score** → cluster separation quality  
- **Cluster balance metric** → avoids dominant + tiny noise clusters  
- **Speaker-count priors** → favors realistic conversations  

The best-scoring configuration is selected automatically.

---

### 🔁 Post-Processing: Speaker Merging
To reduce over-segmentation:
- Compute **centroids** for each speaker cluster
- Measure cosine similarity between centroids
- Merge clusters with very high similarity

This helps handle cases where the same speaker is split into multiple clusters.

---

## 😊 Speech Emotion Recognition (SER)

We integrated a **Speech Emotion Recognition** model to:
- Predict emotion per speech segment
- Align emotional context with speaker labels

This adds an extra semantic layer to the diarization output and enables richer analysis.

---

## 🎨 Frontend (Streamlit)
- Simple, clean interface
- Visualizes:
  - Speaker segments
  - Transcriptions
  - Emotion predictions
- Designed to keep focus on system behavior, not UI complexity

---

## 📈 Results & Learnings

- Adaptive clustering outperformed static thresholds
- Overlap-based segmentation reduced speaker fragmentation
- Explainable heuristics worked better than blind optimization
- Engineering decisions mattered more than model complexity

---

## 🔮 Future Work
- Better handling of overlapping speakers
- Smarter speaker-merging strategies
- Performance and latency optimizations
- Richer visual analytics
- Deeper integration between diarization and emotion recognition

---

## 👥 Team & Acknowledgements

Built with ❤️ by:
- **Anupam**
- **Vansh Saini**
- **Piyush Garg**


