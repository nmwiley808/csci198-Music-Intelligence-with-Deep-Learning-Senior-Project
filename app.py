# =============================================================================
# Music Intelligence with Deep Learning — Demo App
# CSCI 198 Senior Project | Noah Wiley | CSU Fresno
# Supervisor: David Ruby
# =============================================================================
#
# REQUIREMENTS
# Make sure you have the following installed before running:
#   pip install streamlit librosa torch numpy matplotlib pandas
#
# FOLDER STRUCTURE
# Your project folder should look like this:
#   music-intelligence-demo/
#   ├── app.py
#   ├── models/
#   │   ├── cnn_gtzan_best.pth
#   │   ├── cnn_fma_best.pth
#   │   ├── cnn_magna_best.pth
#   │   ├── lstm_gtzan_best.pth
#   │   ├── lstm_fma_best.pth
#   │   ├── lstm_magna_best.pth
#   │   ├── transformer_gtzan_best.pth
#   │   ├── transformer_fma_best.pth
#   │   └── transformer_magna_best.pth
#   └── labels/
#       ├── label_classes_gtzan.npy
#       ├── label_classes_fma.npy
#       └── label_tags_magna.npy
#
# HOW TO RUN
# 1. Open Terminal
# 2. Navigate to your project folder:
#       cd Desktop/music-intelligence-demo
# 3. Run the app:
#       streamlit run app.py
# 4. The app will open automatically in your browser at:
#       http://localhost:8501
#
# HOW TO USE
# 1. Select a dataset from the sidebar (GTZAN, FMA Small, or MagnaTagATune)
# 2. Upload an mp3 or wav audio file
# 3. The app will display:
#    - Audio playback
#    - Log-mel spectrogram visualization
#    - Predictions from all 3 models (CNN, BiLSTM, Transformer)
#    - Confidence scores for each prediction
# =============================================================================

import streamlit as st
import librosa
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import tempfile
import os
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Music Intelligence",
    layout="wide"
)

# Paths
MODELS_DIR = "models"
LABELS_DIR = "labels"

MODEL_PATHS = {
    "GTZAN": {
        "CNN":         os.path.join(MODELS_DIR, "cnn_gtzan_best.pth"),
        "BiLSTM":      os.path.join(MODELS_DIR, "lstm_gtzan_best.pth"),
        "Transformer": os.path.join(MODELS_DIR, "transformer_gtzan_best.pth"),
    },
    "FMA Small": {
        "CNN":         os.path.join(MODELS_DIR, "cnn_fma_best.pth"),
        "BiLSTM":      os.path.join(MODELS_DIR, "lstm_fma_best.pth"),
        "Transformer": os.path.join(MODELS_DIR, "transformer_fma_best.pth"),
    },
    "MagnaTagATune": {
        "CNN":         os.path.join(MODELS_DIR, "cnn_magna_best.pth"),
        "BiLSTM":      os.path.join(MODELS_DIR, "lstm_magna_best.pth"),
        "Transformer": os.path.join(MODELS_DIR, "transformer_magna_best.pth"),
    },
}

LABEL_PATHS = {
    "GTZAN":         os.path.join(LABELS_DIR, "label_classes_gtzan.npy"),
    "FMA Small":     os.path.join(LABELS_DIR, "label_classes_fma.npy"),
    "MagnaTagATune": os.path.join(LABELS_DIR, "label_tags_magna.npy"),
}

# Audio Parameters
SAMPLE_RATE       = 22050
DURATION          = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS            = 128
THRESHOLD         = 0.2

# Model Definitions
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.global_pool(self.features(x)))


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        return self.classifier(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MusicTransformer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, n_layers,
                 dim_feedforward, num_classes, dropout):
        super(MusicTransformer, self).__init__()
        self.input_projection    = nn.Linear(input_size, d_model)
        self.pos_encoding        = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer            = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier          = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# Helper Functions
@st.cache_resource
def load_model(model_name, dataset, num_classes):
    path = MODEL_PATHS[dataset][model_name]
    if model_name == "CNN":
        model = CNN(num_classes=num_classes)
    elif model_name == "BiLSTM":
        if dataset == "MagnaTagATune":
            model = BiLSTM(
                input_size=N_MELS, hidden_size=256,
                num_layers=3, num_classes=num_classes, dropout=0.4
            )
        else:
            model = BiLSTM(
                input_size=N_MELS, hidden_size=128,
                num_layers=2, num_classes=num_classes, dropout=0.3
            )
    else:
        model = MusicTransformer(
            input_size=N_MELS, d_model=128, n_heads=4,
            n_layers=2, dim_feedforward=256,
            num_classes=num_classes, dropout=0.3
        )
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


@st.cache_resource
def load_labels(dataset):
    return np.load(LABEL_PATHS[dataset], allow_pickle=True).tolist()


def preprocess_audio(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
    else:
        y = y[:SAMPLES_PER_TRACK]
    mel     = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return y, log_mel


def predict_cnn(model, log_mel):
    x    = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
    mean = x.mean()
    std  = x.std()
    x    = (x - mean) / (std + 1e-8)
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1).numpy()[0]
    return probs


def predict_sequential(model, log_mel):
    x    = torch.tensor(log_mel.T).unsqueeze(0).float()
    mean = x.mean()
    std  = x.std()
    x    = (x - mean) / (std + 1e-8)
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1).numpy()[0]
    return probs


def predict_magna(model, log_mel, is_cnn=True):
    if is_cnn:
        x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
    else:
        x = torch.tensor(log_mel.T).unsqueeze(0).float()
    mean = x.mean()
    std  = x.std()
    x    = (x - mean) / (std + 1e-8)
    with torch.no_grad():
        out   = model(x)
        probs = torch.sigmoid(out).numpy()[0]
    return probs


def plot_mel_spectrogram(log_mel):
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(log_mel, aspect='auto', origin='lower',
                    cmap='magma', interpolation='nearest')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Log-Mel Spectrogram')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Mel Bins')
    plt.tight_layout()
    return fig


def plot_predictions(probs, labels, model_name, top_n=5):
    top_idx    = np.argsort(probs)[::-1][:top_n]
    top_probs  = probs[top_idx]
    top_labels = [labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(range(top_n), top_probs,
                   color=['#2196F3' if i == 0 else '#90CAF9' for i in range(top_n)])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence')
    ax.set_title(f'{model_name} Predictions')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    return fig


# App Layout
st.title("Music Intelligence with Deep Learning")
st.markdown("**CSCI 198 Senior Project** — Noah Wiley | CSU Fresno")
st.markdown("Upload an audio file to classify it using CNN, BiLSTM, and Transformer models.")
st.divider()

# Sidebar
st.sidebar.title("Settings")
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["GTZAN", "FMA Small", "MagnaTagATune"],
    help="Choose which dataset's model to use for classification"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
if dataset == "GTZAN":
    st.sidebar.markdown("**10 genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock")
elif dataset == "FMA Small":
    st.sidebar.markdown("**8 genres:** Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock")
else:
    st.sidebar.markdown("**Multi-label tagging** — predicts multiple tags per clip (instruments, moods, genres)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
if dataset == "GTZAN":
    st.sidebar.markdown("1. CNN: **77.5%**")
    st.sidebar.markdown("2. Transformer: **61.0%**")
    st.sidebar.markdown("3. BiLSTM: **48.5%**")
elif dataset == "FMA Small":
    st.sidebar.markdown("1. CNN: **59.0%**")
    st.sidebar.markdown("2. Transformer: **52.6%**")
    st.sidebar.markdown("3. BiLSTM: **40.3%**")
else:
    st.sidebar.markdown("1. CNN AUC: **0.8786**")
    st.sidebar.markdown("2. Transformer AUC: **0.8715**")
    st.sidebar.markdown("3. BiLSTM AUC: **0.5003**")

# File Upload
uploaded_file = st.file_uploader(
    "Upload an audio file (mp3 or wav)",
    type=["mp3", "wav"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.markdown("### Uploaded Audio")
    st.audio(uploaded_file)

    with st.spinner("Preprocessing audio..."):
        y_audio, log_mel = preprocess_audio(tmp_path)

    st.markdown("### Log-Mel Spectrogram")
    fig_mel = plot_mel_spectrogram(log_mel)
    st.pyplot(fig_mel)
    plt.close()

    labels     = load_labels(dataset)
    num_labels = len(labels)

    st.markdown("### Model Predictions")

    col1, col2, col3 = st.columns(3)
    model_cols  = [col1, col2, col3]
    model_names = ["CNN", "BiLSTM", "Transformer"]

    for col, model_name in zip(model_cols, model_names):
        with col:
            with st.spinner(f"Running {model_name}..."):
                model = load_model(model_name, dataset, num_labels)

                if dataset == "MagnaTagATune":
                    probs = predict_magna(model, log_mel, is_cnn=(model_name == "CNN"))
                    active_tags = sorted(
                        [(labels[i], probs[i]) for i in range(len(labels)) if probs[i] >= THRESHOLD],
                        key=lambda x: -x[1]
                    )
                    top_tags = sorted(
                        [(labels[i], probs[i]) for i in range(len(labels))],
                        key=lambda x: -x[1]
                    )[:5]

                    st.markdown(f"#### {model_name}")
                    if active_tags:
                        st.markdown("**Tags above threshold:**")
                        for tag, prob in active_tags[:10]:
                            st.progress(float(prob), text=f"{tag} ({prob*100:.1f}%)")
                    else:
                        st.warning("No tags above threshold — showing top 5:")
                        for tag, prob in top_tags:
                            st.progress(float(prob), text=f"{tag} ({prob*100:.1f}%)")
                else:
                    if model_name == "CNN":
                        probs = predict_cnn(model, log_mel)
                    else:
                        probs = predict_sequential(model, log_mel)

                    pred_idx   = probs.argmax()
                    pred_label = labels[pred_idx]
                    confidence = probs[pred_idx] * 100

                    st.markdown(f"#### {model_name}")
                    st.success(f"**{pred_label}** ({confidence:.1f}%)")

                    fig_pred = plot_predictions(probs, labels, model_name)
                    st.pyplot(fig_pred)
                    plt.close()

    os.unlink(tmp_path)

    if dataset != "MagnaTagATune":
        st.markdown("### Prediction Summary")
        summary_data = []
        for model_name in model_names:
            model = load_model(model_name, dataset, num_labels)
            if model_name == "CNN":
                probs = predict_cnn(model, log_mel)
            else:
                probs = predict_sequential(model, log_mel)
            pred_label = labels[probs.argmax()]
            confidence = probs.max() * 100
            summary_data.append({
                "Model":      model_name,
                "Prediction": pred_label,
                "Confidence": f"{confidence:.1f}%",
            })
        st.table(pd.DataFrame(summary_data))

else:
    st.info("Upload an audio file to get started!")

    st.markdown("### Model Performance Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### GTZAN")
        st.markdown("1. CNN: **77.5%**")
        st.markdown("2. Transformer: **61.0%**")
        st.markdown("3. BiLSTM: **48.5%**")

    with col2:
        st.markdown("#### FMA Small")
        st.markdown("1. CNN: **59.0%**")
        st.markdown("2. Transformer: **52.6%**")
        st.markdown("3. BiLSTM: **40.3%**")

    with col3:
        st.markdown("#### MagnaTagATune (AUC)")
        st.markdown("1. CNN: **0.8786**")
        st.markdown("2. Transformer: **0.8715**")
        st.markdown("3. BiLSTM: **0.5003**")