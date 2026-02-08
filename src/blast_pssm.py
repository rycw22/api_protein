# src/blast_pssm.py
import subprocess
from pathlib import Path
import numpy as np

def write_fasta(seq_id: str, seq: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f">{seq_id}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

def run_psiblast_ascii_pssm(fasta_path: Path, db_prefix: Path, out_pssm_path: Path,
                            iters: int = 3, evalue: float = 1e-3, threads: int = 4):
    out_pssm_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "psiblast",
        "-query", str(fasta_path),
        "-db", str(db_prefix),
        "-num_iterations", str(iters),
        "-evalue", str(evalue),
        "-out_ascii_pssm", str(out_pssm_path),
        "-num_threads", str(threads),
    ]

    # capture output so you can detect "No hits found"
    res = subprocess.run(cmd, text=True, capture_output=True)

    # If psiblast outright failed, raise
    if res.returncode != 0:
        raise RuntimeError(f"psiblast failed for {fasta_path.name}\n{res.stderr[:1000]}")

    # If no pssm file was produced (common when no hits), return False
    return out_pssm_path.exists()

def load_ascii_pssm(pssm_path: Path) -> np.ndarray:
    rows = []
    in_matrix = False
    with open(pssm_path, "r") as f:
        for line in f:
            if "Last position-specific scoring matrix" in line:
                in_matrix = True
                continue
            if not in_matrix:
                continue
            parts = line.strip().split()
            if len(parts) >= 22 and parts[0].isdigit() and len(parts[1]) == 1:
                rows.append(list(map(float, parts[2:22])))
            if in_matrix and line.startswith("Lambda"):
                break
    if not rows:
        raise ValueError(f"No PSSM rows found in {pssm_path}")
    pssm = np.array(rows, dtype=np.float32)
    return np.tanh(pssm / 5.0)  # normalized

def get_or_make_pssm(seq_id: str, seq: str, db_prefix: Path, cache_dir: Path,
                     iters: int = 3, evalue: float = 1e-3, threads: int = 4) -> np.ndarray:
    fasta_path = cache_dir / "queries" / f"{seq_id}.fasta"
    pssm_path  = cache_dir / "pssm_ascii" / f"{seq_id}.pssm"

    if not fasta_path.exists():
        write_fasta(seq_id, seq, fasta_path)

    if not pssm_path.exists():
        made = run_psiblast_ascii_pssm(
            fasta_path, db_prefix, pssm_path,
            iters=iters, evalue=evalue, threads=threads
        )
        if not made:
            # fallback: no hits -> return zeros PSSM of correct length
            return np.zeros((len(seq), 20), dtype=np.float32)

    # if it exists, parse it
    return load_ascii_pssm(pssm_path)

