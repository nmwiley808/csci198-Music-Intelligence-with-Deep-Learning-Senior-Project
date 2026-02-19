# Music Intelligence with Deep Learning  
**CSCI 198 – Senior Project**  
**California State University, Fresno**  
**Student: Noah Wiley**  
**Supervisor: Dr. David Ruby**

---

## Repository Overview

This repository contains my senior capstone project for **CSCI 198 – Senior Project** at California State University, Fresno.

The purpose of this project is to explore **music intelligence using deep learning**, focusing on automated music classification tasks such as genre, mood, and instrumentation recognition. The project compares multiple deep learning architectures under a unified experimental framework and evaluates their performance on widely used public music datasets.

---

## Project Description

This project investigates three major deep learning model families:

- Convolutional Neural Networks (CNNs) for spectrogram-based learning  
- Recurrent Neural Networks (LSTMs / BiLSTMs) for temporal modeling  
- Transformer-based architectures for long-range musical structure using attention  

The final outcome includes a research-style comparative analysis and an interactive demo tool that allows users to upload audio and receive predictions.

---

## Objectives

- Preprocess raw audio into suitable feature representations (mel-spectrograms, MFCCs)  
- Implement and train CNN, LSTM, and Transformer models  
- Compare models using standard evaluation metrics  
- Analyze error cases and class overlaps  
- Build a lightweight demo application for real-time inference  
- Produce a formal written report and departmental presentation  

---

## Repository Structure

```
csci198-music-intelligence/
│
├── data/                           # Dataset structure & documentation (no raw audio)
├── src/                            # Preprocessing, models, training, evaluation
├── experiments/                    # Experiment configurations and results
├── notebooks/                      # Exploration and visualization
├── demo/                           # Streamlit/Flask demo application
├── docs/                           # Project plan, report drafts, figures
├── checkpoints/                    # Saved models (ignored by git)
├── requirements.txt                # Project dependencies
├── LICENSE                         # License information
└── README.md                       # Repository Documentation
```

---

## Technologies & Tools

This repository may include:

- Python 3.9+  
- PyTorch / TorchAudio  
- Librosa  
- NumPy / Pandas  
- Scikit-learn  
- Hugging Face Transformers  
- Streamlit or Flask  

---

## Skills Developed

Throughout this project, I am developing skills in:

- Audio signal preprocessing and feature extraction  
- Deep learning model design and training  
- Sequential and attention-based modeling  
- Experimental benchmarking and model comparison  
- Hyperparameter tuning and performance optimization  
- Building interactive AI applications  
- Research-style technical writing and presentation  

---

## Environment Setup

### Install Dependencies

```
pip install -r requirements.txt
```

GPU acceleration is recommended for model training.

---

## Datasets

This project uses publicly available music datasets commonly used in Music Information Retrieval (MIR) research:

- GTZAN – Music genre classification  
- MTG-Jamendo – Multi-label mood and tag prediction  
- MagnaTagATune – Instrument, mood, and genre annotations  

**Note:** Raw audio data is not included in this repository. Users must download datasets separately and comply with each dataset's original license.

---

## Timeline

This project follows a structured 16-week CSCI 198 timeline, including:

- Repository and environment setup  
- Dataset preprocessing  
- Model development and training  
- Evaluation and error analysis  
- Demo application development  
- Final report and departmental presentation  

---

## Deliverables

- Fully documented GitHub repository  
- Trained models and evaluation results  
- Interactive demo application  
- Final research report  
- Department-approved presentation  

---

## License

This project is licensed under the MIT License.  
See the LICENSE file for details.

---

---

## Academic Integrity

This project was completed as part of the requirements for **CSCI 198 – Senior Project** at California State University, Fresno.

All work in this repository is my own unless otherwise stated. Any external libraries, datasets, or research references are properly credited.

This repository is maintained for academic, research, and portfolio documentation purposes only.
