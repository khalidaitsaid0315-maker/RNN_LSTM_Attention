"""
RNN/LSTM Interface - Prediction de series financieres.

Version simple de l'application Streamlit pour comparer Ridge, MLP-RNN,
MLP-LSTM et un modele avec attention manuelle.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from utils import load_npy_data, validate_data_shapes

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="RNN/LSTM Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5em;
        color: #ff7f0e;
        margin-top: 1.5em;
    }
    .success-box {
        background: #d4edda;
        padding: 12px;
        border-radius: 8px;
    }
    .info-box {
        background: #d1ecf1;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def generate_sample_data(n_samples=500, seq_len=60, n_features=18):
    """Genere des donnees de series temporelles synthetiques."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, seq_len, n_features))
    y = np.mean(X[:, -5:, :5], axis=(1, 2)) + rng.normal(scale=0.05, size=n_samples)
    return X, y


def flatten_sequences(X):
    """Aplatit les sequences (samples, T, F) vers (samples, T*F)."""
    return X.reshape(X.shape[0], -1)


def calculate_metrics(predictions, targets):
    """Calcule MSE, RMSE et MAE."""
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


def prepare_dataset(data_source, data_dir, n_samples, seq_len, n_features, test_size):
    """Prepare train/test arrays from synthetic data or .npy files."""
    if data_source == "Fichiers .npy":
        X_train, y_train, X_test, y_test = load_npy_data(data_dir)
        validate_data_shapes(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test

    X, y = generate_sample_data(n_samples, seq_len, n_features)
    split_idx = int(len(X) * (1 - test_size / 100))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def softmax(x, axis=-1):
    """Softmax numeriquement stable."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class SimpleAttentionModel:
    """Modele avec mecanisme d'attention manuel."""

    def __init__(self, attention_dim=32, alpha=1.0):
        self.attention_dim = attention_dim
        self.alpha = alpha
        self.Wa = None
        self.v = None
        self.ridge = Ridge(alpha=alpha)
        self.weights = None

    def _init_weights(self, n_features):
        rng = np.random.default_rng(42)
        self.Wa = rng.normal(scale=0.1, size=(n_features, self.attention_dim))
        self.v = rng.normal(scale=0.1, size=self.attention_dim)

    def fit(self, X, y):
        self._init_weights(X.shape[2])
        scores = np.tanh(X @ self.Wa) @ self.v
        weights = softmax(scores, axis=1)
        context = np.einsum("st,stf->sf", weights, X)
        self.ridge.fit(context, y)
        self.weights = weights
        return self

    def predict(self, X):
        scores = np.tanh(X @ self.Wa) @ self.v
        weights = softmax(scores, axis=1)
        context = np.einsum("st,stf->sf", weights, X)
        return self.ridge.predict(context)


st.markdown("<div class='main-header'>RNN/LSTM Interface</div>", unsafe_allow_html=True)
st.markdown("Comparaison de modeles pour la prediction de series financieres")
st.markdown("---")


with st.sidebar:
    st.markdown("### Configuration")

    data_source = st.radio(
        "Source des donnees",
        ["Synthetiques", "Fichiers .npy"],
        help="Utilisez des donnees generees automatiquement ou chargez des fichiers .npy.",
    )

    st.subheader("Donnees")
    if data_source == "Synthetiques":
        n_samples = st.slider("Nombre d'echantillons", 100, 1000, 300, step=100)
        seq_len = st.slider("Longueur de sequence", 10, 100, 40, step=10)
        n_features = st.slider("Nombre de features", 5, 50, 15, step=5)
        test_size = st.slider("Taille du test (%)", 10, 40, 20, step=5)
        data_dir = "./data"
    else:
        data_dir = st.text_input("Dossier des .npy", value="./data")
        n_samples, seq_len, n_features, test_size = 500, 60, 18, 20
        st.caption("Fichiers attendus : X_train.npy, y_train.npy, X_test.npy, y_test.npy")

    st.subheader("Hyperparametres")
    ridge_alpha = st.slider("Ridge alpha", 0.1, 10.0, 1.0)
    mlp_hidden_size = st.selectbox(
        "Couches cachees MLP",
        [(64, 32), (128, 64), (256, 128, 64)],
    )
    attention_dim = st.slider("Attention dimension", 8, 64, 32, step=8)
    max_iter = st.slider("Iterations MLP", 50, 300, 120, step=50)

    run_button = st.button("Executer l'analyse", use_container_width=True)


if run_button:
    with st.spinner("Preparation des donnees..."):
        try:
            X_train, y_train, X_test, y_test = prepare_dataset(
                data_source, data_dir, n_samples, seq_len, n_features, test_size
            )
        except Exception as exc:
            st.error(f"Impossible de preparer les donnees : {exc}")
            st.stop()

        X_train_flat = flatten_sequences(X_train)
        X_test_flat = flatten_sequences(X_test)

    with st.spinner("Entrainement des modeles..."):
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_train_flat, y_train)
        ridge_preds = ridge.predict(X_test_flat)
        ridge_metrics = calculate_metrics(ridge_preds, y_test)

        mlp_rnn = MLPRegressor(
            hidden_layer_sizes=mlp_hidden_size,
            activation="tanh",
            max_iter=max_iter,
            early_stopping=True,
            random_state=42,
            verbose=0,
        )
        mlp_rnn.fit(X_train_flat, y_train)
        mlp_rnn_preds = mlp_rnn.predict(X_test_flat)
        mlp_rnn_metrics = calculate_metrics(mlp_rnn_preds, y_test)

        mlp_lstm = MLPRegressor(
            hidden_layer_sizes=tuple(x * 2 for x in mlp_hidden_size),
            activation="relu",
            max_iter=max_iter,
            early_stopping=True,
            random_state=42,
            verbose=0,
        )
        mlp_lstm.fit(X_train_flat, y_train)
        mlp_lstm_preds = mlp_lstm.predict(X_test_flat)
        mlp_lstm_metrics = calculate_metrics(mlp_lstm_preds, y_test)

        attn_model = SimpleAttentionModel(attention_dim=attention_dim, alpha=ridge_alpha)
        attn_model.fit(X_train, y_train)
        attn_preds = attn_model.predict(X_test)
        attn_metrics = calculate_metrics(attn_preds, y_test)

    st.session_state["results"] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preds": {
            "Ridge": ridge_preds,
            "MLP-RNN": mlp_rnn_preds,
            "MLP-LSTM": mlp_lstm_preds,
            "RNN+Attention": attn_preds,
        },
        "metrics": {
            "Ridge": ridge_metrics,
            "MLP-RNN": mlp_rnn_metrics,
            "MLP-LSTM": mlp_lstm_metrics,
            "RNN+Attention": attn_metrics,
        },
        "attn_weights": attn_model.weights,
    }

if "results" not in st.session_state:
    st.info(
        "Configurez les parametres dans la barre laterale, puis cliquez sur "
        "\"Executer l'analyse\" pour lancer les modeles."
    )
    st.markdown(
        """
        ### Modeles disponibles

        1. Ridge Regression
        2. MLP-RNN avec activation tanh
        3. MLP-LSTM avec activation relu
        4. RNN avec mecanisme d'attention manuel
        """
    )
    st.stop()


results = st.session_state["results"]
X_train = results["X_train"]
X_test = results["X_test"]
y_test = results["y_test"]
preds = results["preds"]
metrics = results["metrics"]
metrics_df = pd.DataFrame(metrics).T[["MSE", "RMSE", "MAE"]]
models_names = list(metrics.keys())
colors = ["#808080", "#1f77b4", "#ff7f0e", "#2ca02c"]


st.markdown("<div class='sub-header'>Resultats</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
for col, name in zip([col1, col2, col3, col4], models_names):
    with col:
        st.markdown(f"### {name}")
        st.metric("MSE", f"{metrics[name]['MSE']:.6f}")
        st.metric("RMSE", f"{metrics[name]['RMSE']:.6f}")
        st.metric("MAE", f"{metrics[name]['MAE']:.6f}")

st.dataframe(metrics_df.style.highlight_min(axis=0, color="#d4edda"), use_container_width=True)
st.download_button(
    "Telecharger les metriques CSV",
    metrics_df.to_csv(index_label="model").encode("utf-8"),
    file_name="rnn_lstm_metrics.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("---")
st.markdown("### Comparaison des Modeles")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric_key in zip(axes, ["MSE", "RMSE", "MAE"]):
    vals = metrics_df[metric_key].values
    bars = ax.bar(models_names, vals, color=colors)
    ax.set_title(metric_key, fontweight="bold", fontsize=12)
    ax.set_ylabel("Valeur")
    ax.grid(axis="y", alpha=0.3)
    best_idx = int(np.argmin(vals))
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(2.5)
    ax.tick_params(axis="x", rotation=12)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("### Predictions vs Valeurs Reelles")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, name, color in zip(axes.flat, models_names, colors):
    n_plot = min(100, len(y_test))
    ax.plot(y_test[:n_plot], label="Reel", color="black", linewidth=1.5)
    ax.plot(preds[name][:n_plot], label="Predit", color=color, alpha=0.8)
    ax.set_title(name, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("### Distribution des Erreurs")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, name, color in zip(axes.flat, models_names, colors):
    errors = preds[name] - y_test
    ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{name} - Residus", fontweight="bold")
    ax.set_xlabel("Erreur")
    ax.grid(alpha=0.3, axis="y")
    ax.text(
        0.05,
        0.90,
        f"mu={errors.mean():.4f}\nsigma={errors.std():.4f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("### Analyse Detaillee")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, name, color in zip(axes.flat, models_names, colors):
    ax.scatter(y_test, preds[name], alpha=0.3, s=20, color=color)
    lims = [min(y_test.min(), preds[name].min()), max(y_test.max(), preds[name].max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Parfait")
    ax.set_xlabel("Valeur reelle")
    ax.set_ylabel("Valeur predite")
    ax.set_title(f"{name} - Reel vs Predit", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("---")
st.markdown("<div class='sub-header'>Interpretations</div>", unsafe_allow_html=True)

best_model_name = metrics_df["RMSE"].idxmin()
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div class='success-box'>
        <h4>Meilleur Modele</h4>
        <p><strong>{best_model_name}</strong> avec RMSE = {metrics_df.loc[best_model_name, 'RMSE']:.6f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    improvement = (
        (metrics["MLP-RNN"]["RMSE"] - metrics["RNN+Attention"]["RMSE"])
        / metrics["MLP-RNN"]["RMSE"]
        * 100
    )
    st.markdown(
        f"""
        <div class='info-box'>
        <h4>Amelioration</h4>
        <p>MLP-RNN vers RNN+Attention : <strong>{improvement:+.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    ### Guide des Modeles

    **Ridge (Baseline)** : modele lineaire simple et rapide.

    **MLP-RNN (tanh)** : reseau multicouche avec activation tanh.

    **MLP-LSTM (relu)** : reseau plus profond avec activation relu.

    **RNN+Attention (Manuel)** : pondere chaque pas de temps selon son importance.
    """
)

st.success("Analyse complete. Ajustez les parametres et relancez pour tester.")

st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    Projet : Prediction de Series Financieres avec RNN/LSTM<br/>
    Developpe avec Streamlit, NumPy et scikit-learn
</div>
""",
    unsafe_allow_html=True,
)
