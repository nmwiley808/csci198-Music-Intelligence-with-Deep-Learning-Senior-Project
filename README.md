# Music Intelligence with Deep Learning

**CSCI 198 - Senior Project**
Department of Computer Sciecne, California State University, Fresno

**Student:** Noah Wiley

**Supervisor:** David Ruby

---

## Project Overview

This project explores **music intelligence using deep learning**, focusing on automated music classification tasks such as **genre, mood, and instrumentation recognition**. The goal is to design, train, and compare multiple deep learning architectures under a unified experiment framework, and to evaluate their performance on widely used public music datasets.

This project investigates three major model famiilies:
- **Conbolutional Neural Networks (CNNs)** for spectrogram-based learning
- **Recurrent Neural Networks (LSTMs)** for temporal modeling
- **Transformer-based architectures** for long-range musical structure via attention

The fial outcome includes a research-style comparative analysis and an  interactive demo tool that allows users to upload audio and receive predictions.

---

## Objectives

- Preprocess raw audio into suitable feature representations (mel-spectrograms, MFCCs)
- Implement and train CNN, LSTM, and Transformer models
- Compare models using standard evaluation metrics
- Analyze error cases and class overlaps
- Build a lightweight demo application for real-time inference
- Produce a formal written report and departmental presentation

---

## Datasets

This project uses publicly available music datasets commonly used in Music Information Retrieval (MIR) research:

- **GTZAN** - music genre classification
- **MTG-Jamendo** - multi-label mood and tag prediction
- **MagnaTagATune** - instrument, mood, and genre annotations

**Note:** Raw audio data is not included in this repository. Users must download datasets separately and comply with each dataset's original license.

---

## Methods & Models

### Audio Representations
- Mel-spectrograms
- Log-mel features
- MFCCs

### Model Architectures
- CNNs for spatial feature extraction from spectrograms
- LSTMs / BiLSTMs for sequential audio modeling
- Transformers with positional encodings and self-attention

### Evaluation Metrics
- Accuracy
- Precision / Recall
- F1 Score
- Confusion Matrices
- Training & Validation Curves

---

## Repository Structure

csci198-music-intelligence/

|  data/ # Dataset structure & documentation (no raw audio)

|  src/  # Preprocessing, models, training, evaluation

|  experiments/ # Experiment configs and results

|  notebooks/ # Exploration and visualization

|  demo/ # Streamlit/Flask demo app

|  docs/ # Project plan, report drafts, figures

|  checkpoints/ # Saved models (ignored by git)

|  requirements.txt

|  README.md

|  LICENSE

---

## Enviroment Setup

### Python
- Python 3.9+ recommended

### Install Dependencies
'''bash
pip install -r requirements.txt

---

## Key Libraries:
- PyTorch/ TorchAudio
- Librosa
- NumPy, Pandas
- Scikit-learn
- Transformers
- Streamlit

---

# Timeline
This project follows a 16-week CSCI 198 timeline, including:
- Repository & enviroment setup
- Dataset preprocessing
- Model development & training
- Evalution & error analysis
- Demo application development
- Final report & presentation

---

# Deliverables
- Fully documented GitHub repository
- Trained models and evalutaion results
- Interactive demo application
- Final research report
- Department-approved presentation

---

# License
This project is licensed under the MIT License
See the LICENSE file for details
