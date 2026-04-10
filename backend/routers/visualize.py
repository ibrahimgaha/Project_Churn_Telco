"""
routers/visualize.py
====================
Dimensionality reduction endpoints for data exploration.

PCA (Principal Component Analysis):
  Linear projection onto the 2 directions of maximum variance.
  Fast and deterministic. Good for seeing broad separability.

t-SNE (t-distributed Stochastic Neighbor Embedding):
  Non-linear method that preserves local structure.
  Slower but often reveals clusters that PCA misses.

Both return 2D points coloured by the Churn label.
"""

from fastapi import APIRouter, Query
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import load_dataset, encode_features

router = APIRouter()


def _prepare_2d(method: str, perplexity: int = 30):
    """Shared logic: load data → encode → reduce to 2D → return JSON."""
    df = load_dataset()
    df_encoded, _ = encode_features(df)

    target_col = "Churn"
    features = [c for c in df_encoded.columns if c != target_col]

    X = df_encoded[features].values
    y = df_encoded[target_col].values

    # Always scale before dim-reduction for fair distance computation
    X_scaled = StandardScaler().fit_transform(X)

    if method == "pca":
        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    else:
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            n_iter=800,
        ).fit_transform(X_scaled)

    # Build response: list of {x, y, label}
    points = [
        {"x": round(float(coords[i, 0]), 4),
         "y": round(float(coords[i, 1]), 4),
         "label": int(y[i])}
        for i in range(len(y))
    ]
    return {"method": method, "points": points, "total": len(points)}


@router.get("/pca")
def get_pca():
    """Return 2D PCA projection of the full dataset."""
    return _prepare_2d("pca")


@router.get("/tsne")
def get_tsne(perplexity: int = Query(30, ge=5, le=50)):
    """
    Return 2D t-SNE projection.
    perplexity controls the balance between local and global structure.
    """
    return _prepare_2d("tsne", perplexity=perplexity)
