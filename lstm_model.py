# lstm_model.py — ActionLSTM + loader + predict()
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Your original model
# =========================
class ActionLSTM(nn.Module):
    def __init__(self, input_size=150, hidden_size=256, num_layers=3, num_classes=4):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        """
        x: [batch, seq_len, input_size] where input_size=150 (e.g., flattened 2×(25*3) or similar)
        """
        lstm_out, _ = self.lstm(x)               # [B, T, 2H]
        attn_weights = self.attention(lstm_out)  # [B, T, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, 2H]
        logits = self.fc(context)                # [B, C]
        return logits

# =========================
# New: config + convenience API
# =========================
# EDIT this to your exact label order used during training:
CLASS_NAMES = ["walk", "fall", "stand", "jump"]

# Default weights filename (put it next to this file)
DEFAULT_WEIGHTS = "action_lstm.pth"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model  = None  # filled by load_model()

# def load_model(weights_path: str = DEFAULT_WEIGHTS,
#                input_size: int = 150,
#                hidden_size: int = 256,
#                num_layers: int = 3,
#                num_classes: int = None):
#     """
#     Create & load an ActionLSTM and keep it in a module-global `_model`.
#     """
#     global _model
#     if num_classes is None:
#         num_classes = len(CLASS_NAMES)
#     m = ActionLSTM(input_size=input_size,
#                    hidden_size=hidden_size,
#                    num_layers=num_layers,
#                    num_classes=num_classes)
#     m.to(_DEVICE)
#     if weights_path and os.path.exists(weights_path):
#         state = torch.load(weights_path, map_location=_DEVICE)
#         # support both full checkpoints ({'state_dict': ...}) and raw state_dict
#         state_dict = state.get("state_dict", state)
#         m.load_state_dict(state_dict, strict=True)
#         print(f"[lstm_model] loaded weights: {weights_path}")
#     else:
#         print(f"[lstm_model] WARNING: weights file not found: {weights_path}")
#     m.eval()
#     _model = m
#     return _model

def load_model(weights_path=None):
    global _model
    _model = ActionLSTM()
    _model.eval()

    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        _model.load_state_dict(state)
        print(f"[model] loaded weights from {weights_path}")
    else:
        print("[model] no weights found, using random init")  # 👈 new line
        # leave model with random weights

    return _model


def _ensure_model():
    global _model
    if _model is None:
        # lazy-load with defaults
        load_model(DEFAULT_WEIGHTS)
    return _model

def preprocess_window(seq_ntu):
    """
    seq_ntu: list/array of length T, each item shape (25,3) in normalized coords.
    Returns x: np.ndarray with shape [1, T, 150] matching ActionLSTM input_size=150.

    We assume training used 150-dim per frame. If you originally had 2 bodies,
    we place body-A (25*3=75 dims) then body-B (75 dims). Since we track one
    person here, we zero-pad the second 75 dims.
    """
    arr = np.stack(seq_ntu, axis=0)  # [T,25,3]
    T = arr.shape[0]
    # body A flattened:
    flatA = arr.reshape(T, -1)  # [T, 75]
    # body B zeros:
    flatB = np.zeros_like(flatA)  # [T, 75]
    flat150 = np.concatenate([flatA, flatB], axis=1)  # [T, 150]
    return flat150.reshape(1, T, 150).astype(np.float32)

@torch.no_grad()
def predict(x):
    m = _ensure_model()
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float32)

    if x.ndim != 3:
        raise ValueError(f"predict expects [1, T, F]; got shape {x.shape}")

    B, T, F = x.shape
    if F == 75:
        zeros = np.zeros((B, T, 75), dtype=x.dtype)
        x = np.concatenate([x, zeros], axis=-1)
        F = 150
    if F != 150:
        raise ValueError(f"input_size must be 150 (got {F}). Adjust preprocess to match training.")

    tens = torch.from_numpy(x).to(_DEVICE).float()
    logits = m(tens)
    # 👇 critical change: no .numpy() here
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

