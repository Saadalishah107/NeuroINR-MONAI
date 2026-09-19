"""
src/quantum/quantum_feature_map.py

EXPLORATORY EXTENSION -- not part of the core diagnostic pipeline.

Implements a classically-simulated quantum feature map in the style of
Havlicek et al. 2019 ("Supervised learning with quantum-enhanced feature
spaces"): each input feature is angle-encoded onto a qubit
(|psi> = cos(theta/2)|0> + sin(theta/2)|1>), followed by a fixed
entangling layer (ZZ-type phase coupling between qubit pairs), producing
a higher-dimensional embedding via the resulting statevector amplitudes.
Everything here is standard linear algebra over complex numpy arrays --
there is no quantum hardware or quantum simulator library involved, and
this module says so explicitly rather than implying otherwise.

Use in this project: applied to the CNN's 2-4-dimensional PCA-reduced
penultimate-layer embeddings (src/stats/matrix_stats.py) as an alternative
feature space, then a plain logistic regression is trained on top of it
and compared against the same classifier trained on the raw PCA features.
This tests one narrow, honest question -- "does this particular quantum-
inspired embedding change linear separability on this dataset?" -- not
"quantum computers improve MRI diagnosis," which this experiment cannot
and does not claim.
"""
from __future__ import annotations

import numpy as np


def _rotation_state(theta: float) -> np.ndarray:
    """Single-qubit state after an Ry(theta) rotation from |0>."""
    return np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128)


def _kron_all(states):
    out = states[0]
    for s in states[1:]:
        out = np.kron(out, s)
    return out


def _zz_entangle(state: np.ndarray, n_qubits: int, coupling: float = np.pi / 4) -> np.ndarray:
    """Applies a fixed nearest-neighbour ZZ phase coupling exp(i*coupling*Z_k Z_{k+1})
    to the statevector -- a standard, data-independent entangling layer used
    in quantum feature maps to create feature interactions that a purely
    classical angle encoding (no entanglement) cannot represent."""
    dim = 2 ** n_qubits
    phases = np.ones(dim, dtype=np.complex128)
    for idx in range(dim):
        bits = [(idx >> b) & 1 for b in range(n_qubits)]
        z_vals = [1 - 2 * b for b in bits]  # bit 0 -> Z=+1, bit 1 -> Z=-1
        for k in range(n_qubits - 1):
            phases[idx] *= np.exp(1j * coupling * z_vals[k] * z_vals[k + 1])
    return state * phases


def quantum_feature_embedding(x: np.ndarray, entangle: bool = True) -> np.ndarray:
    """
    x: (n_features,) array of real values, expected roughly in [-1, 1]
       (features outside that range still work, they just saturate cos/sin).
    Returns: (2, ) squared-amplitude probability vector per computational
    basis outcome pair -- concretely, we return |psi|^2 over the full
    2^n_qubits statevector, giving a (2**n_features,) embedding.
    """
    n = len(x)
    thetas = np.pi * np.clip(x, -1, 1)  # map feature range to a rotation angle
    single_qubit_states = [_rotation_state(t) for t in thetas]
    state = _kron_all(single_qubit_states)
    if entangle and n > 1:
        state = _zz_entangle(state, n)
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()  # renormalise for numerical safety
    return probs.astype(np.float64)


def batch_quantum_embedding(X: np.ndarray, entangle: bool = True) -> np.ndarray:
    """X: (N, n_features) -> (N, 2**n_features)"""
    return np.stack([quantum_feature_embedding(x, entangle=entangle) for x in X])


def compare_classical_vs_quantum_features(X_train, y_train, X_test, y_test, entangle=True):
    """
    Trains a plain logistic regression on (a) the raw/PCA'd classical
    features and (b) the quantum-embedded features, and reports AUROC/AUPRC
    for both -- the honest exploratory comparison this module exists for.
    """
    from sklearn.linear_model import LogisticRegression
    from src.utils.metrics import classification_report_dict

    clf_classical = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    p_classical = clf_classical.predict_proba(X_test)[:, 1]
    report_classical = classification_report_dict(y_test, p_classical)

    Xq_train = batch_quantum_embedding(X_train, entangle=entangle)
    Xq_test = batch_quantum_embedding(X_test, entangle=entangle)
    clf_quantum = LogisticRegression(max_iter=2000).fit(Xq_train, y_train)
    p_quantum = clf_quantum.predict_proba(Xq_test)[:, 1]
    report_quantum = classification_report_dict(y_test, p_quantum)

    return {"classical": report_classical, "quantum_embedded": report_quantum}
