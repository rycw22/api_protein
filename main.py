# main.py
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- Import your PSSM helper (same as your training setup) ----
# Make sure your project has: src/blast_pssm.py
from src.blast_pssm import get_or_make_pssm


# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI(title="Protein Helix API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for testing; tighten later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Constants copied from your rnn.py logic
# -------------------------
MAX_LEN = 512
AMINO_ACIDS = list("acdefghiklmnpqrstvwy")  # 20
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 1..20
UNK_IDX = len(AA_TO_IDX) + 1  # 21

# Biophysical tables
KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
CHARGE = {aa: 0.0 for aa in list("ACDEFGHIKLMNPQRSTVWY")}
CHARGE.update({"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0})

HELIX_PROP = {
    "A": 1.45, "C": 0.77, "D": 1.01, "E": 1.51, "F": 1.13,
    "G": 0.53, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
    "M": 1.45, "N": 0.67, "P": 0.59, "Q": 1.11, "R": 0.98,
    "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}
STERIC_BULK = {
    "A": 88.6,  "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1,  "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0,  "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

_bulk_vals = np.array(list(STERIC_BULK.values()), dtype=np.float32)
BULK_MEAN = float(_bulk_vals.mean())
BULK_STD = float(_bulk_vals.std() if _bulk_vals.std() > 0 else 1.0)

HYDRO_BY_IDX = np.zeros(UNK_IDX + 1, dtype=np.float32)
CHARGE_BY_IDX = np.zeros(UNK_IDX + 1, dtype=np.float32)
HELIX_BY_IDX = np.zeros(UNK_IDX + 1, dtype=np.float32)
BULK_BY_IDX = np.zeros(UNK_IDX + 1, dtype=np.float32)

for i, aa in enumerate(AMINO_ACIDS, start=1):
    AA = aa.upper()
    HYDRO_BY_IDX[i] = KD_SCALE[AA]
    CHARGE_BY_IDX[i] = CHARGE[AA]
    HELIX_BY_IDX[i] = HELIX_PROP[AA]
    BULK_BY_IDX[i] = (STERIC_BULK[AA] - BULK_MEAN) / BULK_STD


# -------------------------
# PSSM / BLAST configuration (env overridable)
# -------------------------
DB_PREFIX = Path(os.environ.get("BLAST_DB_PREFIX", "db/swissprot"))
CACHE_DIR = Path(os.environ.get("PSSM_CACHE_DIR", "cache"))
PSIBLAST_ITERS = int(os.environ.get("PSIBLAST_ITERS", "3"))
PSIBLAST_EVALUE = float(os.environ.get("PSIBLAST_EVALUE", "0.001"))
PSIBLAST_THREADS = int(os.environ.get("PSIBLAST_THREADS", "2"))


# -------------------------
# Helpers
# -------------------------
def pad_1d(values: List[int], max_len: int, pad_value: int = 0) -> Tuple[np.ndarray, int]:
    out = np.full((max_len,), pad_value, dtype=np.int32)
    L = min(max_len, len(values))
    out[:L] = np.asarray(values[:L], dtype=np.int32)
    return out, L

def onehot_no_pad(X_tokens: np.ndarray) -> np.ndarray:
    # X_tokens: (N, L) values in [0..UNK_IDX]
    N, L = X_tokens.shape
    out = np.zeros((N, L, UNK_IDX), dtype=np.float32)  # 21 channels
    mask = X_tokens != 0
    out[mask, X_tokens[mask] - 1] = 1.0
    return out

def biophys_from_tokens(X_tokens: np.ndarray) -> np.ndarray:
    h = HYDRO_BY_IDX[X_tokens][..., None]
    c = CHARGE_BY_IDX[X_tokens][..., None]
    hp = HELIX_BY_IDX[X_tokens][..., None]
    b = BULK_BY_IDX[X_tokens][..., None]
    return np.concatenate([h, c, hp, b], axis=-1).astype(np.float32)

def pad_pssm(pssm: np.ndarray, max_len: int) -> np.ndarray:
    out = np.zeros((max_len, 20), dtype=np.float32)
    L = min(max_len, pssm.shape[0])
    out[:L] = pssm[:L]
    return out

def clean_sequence(seq: str) -> str:
    # Remove FASTA headers and whitespace; keep letters only
    lines = [ln.strip() for ln in seq.splitlines() if ln.strip()]
    if lines and lines[0].startswith(">"):
        lines = [ln for ln in lines if not ln.startswith(">")]
    joined = "".join(lines)
    # Keep A-Z only
    joined = "".join(ch for ch in joined.upper() if "A" <= ch <= "Z")
    return joined

def build_features_for_sequence(seq: str, seq_id: str) -> Tuple[np.ndarray, int]:
    seq = clean_sequence(seq)
    if not seq:
        raise ValueError("Empty sequence after cleaning. Paste amino-acid letters (optionally FASTA).")

    tokens = [AA_TO_IDX.get(ch.lower(), UNK_IDX) for ch in seq]
    tok_pad, L = pad_1d(tokens, MAX_LEN, pad_value=0)
    X_tokens = tok_pad[None, :]  # (1, MAX_LEN)

    X_id = onehot_no_pad(X_tokens)         # (1, MAX_LEN, 21)
    X_bio = biophys_from_tokens(X_tokens)  # (1, MAX_LEN, 4)

    # PSSM: (len(seq_trunc), 20)
    seq_trunc = seq[:MAX_LEN]
    try:
        pssm = get_or_make_pssm(
            seq_id=seq_id,
            seq=seq_trunc,
            db_prefix=DB_PREFIX,
            cache_dir=CACHE_DIR,
            iters=PSIBLAST_ITERS,
            evalue=PSIBLAST_EVALUE,
            threads=PSIBLAST_THREADS,
        )
        if not isinstance(pssm, np.ndarray) or pssm.ndim != 2 or pssm.shape[1] != 20:
            raise ValueError("PSSM returned in unexpected shape.")
        if pssm.shape[0] != len(seq_trunc):
            raise ValueError("PSSM length mismatch.")
    except Exception:
        # If BLAST/PSSM isn't set up, fall back to zeros (still lets API work for demos)
        pssm = np.zeros((len(seq_trunc), 20), dtype=np.float32)

    X_pssm = pad_pssm(pssm, MAX_LEN)[None, :, :]  # (1, MAX_LEN, 20)
    X_feat = np.concatenate([X_id, X_pssm, X_bio], axis=-1)  # (1, MAX_LEN, 45)
    return X_feat, min(L, MAX_LEN)

def helix_regions_from_probs(helix_probs: np.ndarray, L: int, threshold: float) -> List[Tuple[int, int, float]]:
    # Returns list of (start, end_exclusive, mean_conf)
    regions: List[Tuple[int, int, float]] = []
    i = 0
    while i < L:
        if float(helix_probs[i]) >= threshold:
            start = i
            confs: List[float] = []
            while i < L and float(helix_probs[i]) >= threshold:
                confs.append(float(helix_probs[i]))
                i += 1
            end = i
            regions.append((start, end, float(np.mean(confs)) if confs else 0.0))
        else:
            i += 1
    return regions


# -------------------------
# Load model once
# -------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "helix_model_with_pssm_biophys_hp_bulk.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# -------------------------
# API Schemas
# -------------------------
class PredictionRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence (plain letters or FASTA).")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Helix confidence threshold.")
    # Keep this optional so your frontend can send it without breaking anything
    windowSize: Optional[int] = Field(None, ge=1, le=512, description="(Optional) UI window size; not used in model inference.")

    class Config:
        extra = "allow"  # allow extra fields without 422s


class HelixResult(BaseModel):
    id: str
    sequence: str
    confidence: float
    classification: str
    name: str
    start: int
    end: int


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        seq_clean = clean_sequence(req.sequence)
        X_feat, L = build_features_for_sequence(seq_clean, seq_id="api_query")

        probs = model.predict(X_feat, verbose=0)  # expected (1, MAX_LEN, 3)
        if not isinstance(probs, np.ndarray) or probs.ndim != 3 or probs.shape[0] != 1:
            raise RuntimeError(f"Unexpected model output shape: {getattr(probs, 'shape', None)}")

        # Convention from your training: 0=coil, 1=helix, 2=strand
        helix_probs = probs[0, :, 1]  # (MAX_LEN,)

        regions = helix_regions_from_probs(helix_probs, L, float(req.threshold))
        helices: List[dict] = []
        for k, (s, e, conf) in enumerate(regions, start=1):
            subseq = seq_clean[s:e]
            helices.append({
                "id": f"helix_{k}",
                "sequence": subseq,
                "confidence": conf,
                "classification": "Alpha Helix",
                "name": f"Helix Region {k} ({s}-{e-1})",
                "start": s,
                "end": e - 1,
            })

        return {"helices": helices}

    except ValueError as e:
        # Input/validation-ish errors
        print("Predict ValueError:", repr(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected server/model errors
        print("Predict error:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
