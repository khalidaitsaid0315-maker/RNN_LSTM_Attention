"""Streamlit app for comparing simple time-series regression models."""

import traceback

import streamlit as st


st.set_page_config(
    page_title="RNN/LSTM Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_style():
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 2.4rem;
                color: #1f77b4;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .sub-header {
                font-size: 1.35rem;
                color: #ff7f0e;
                font-weight: 700;
                margin-top: 1.4rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_dependencies():
    import numpy as np

    return np


def generate_sample_data(np, n_samples=300, seq_len=40, n_features=15):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, seq_len, n_features))
    y = np.mean(X[:, -5:, :5], axis=(1, 2)) + rng.normal(scale=0.05, size=n_samples)
    return X, y


def load_npy_dataset(np, data_dir):
    from pathlib import Path

    data_path = Path(data_dir)
    required = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    missing = [name for name in required if not (data_path / name).exists()]
    if missing:
        raise FileNotFoundError("Fichiers manquants : " + ", ".join(missing))

    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("X_train et X_test doivent etre 3D : samples, timesteps, features")
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("y_train et y_test doivent etre 1D")
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("X_train et X_test doivent avoir les memes dimensions sequence/features")
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Le nombre de samples X/y ne correspond pas")

    return X_train, y_train, X_test, y_test


def split_synthetic_data(np, n_samples, seq_len, n_features, test_size):
    X, y = generate_sample_data(np, n_samples, seq_len, n_features)
    split_idx = int(len(X) * (1 - test_size / 100))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)


def calculate_metrics(np, predictions, targets):
    mse = float(np.mean((predictions - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(predictions - targets)))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


def softmax(np, x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def add_bias(np, X):
    return np.c_[np.ones(X.shape[0]), X]


def ridge_predict(np, X_train, y_train, X_test, alpha):
    Xb_train = add_bias(np, X_train)
    Xb_test = add_bias(np, X_test)
    reg = alpha * np.eye(Xb_train.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(Xb_train.T @ Xb_train + reg) @ Xb_train.T @ y_train
    return Xb_test @ coef


def random_feature_predict(np, X_train, y_train, X_test, hidden_layers, alpha, activation):
    rng = np.random.default_rng(42)
    features_train = X_train
    features_test = X_test

    for layer_size in hidden_layers:
        scale = 1.0 / np.sqrt(max(features_train.shape[1], 1))
        W = rng.normal(scale=scale, size=(features_train.shape[1], layer_size))
        b = rng.normal(scale=0.1, size=layer_size)
        z_train = features_train @ W + b
        z_test = features_test @ W + b

        if activation == "tanh":
            features_train = np.tanh(z_train)
            features_test = np.tanh(z_test)
        else:
            features_train = np.maximum(0.0, z_train)
            features_test = np.maximum(0.0, z_test)

    return ridge_predict(np, features_train, y_train, features_test, alpha)


class SimpleAttentionModel:
    def __init__(self, np, attention_dim=32, alpha=1.0):
        self.np = np
        self.attention_dim = attention_dim
        self.alpha = alpha
        self.Wa = None
        self.v = None
        self.context_train = None
        self.y_train = None
        self.weights = None

    def _init_weights(self, n_features):
        rng = self.np.random.default_rng(42)
        self.Wa = rng.normal(scale=0.1, size=(n_features, self.attention_dim))
        self.v = rng.normal(scale=0.1, size=self.attention_dim)

    def fit(self, X, y):
        self._init_weights(X.shape[2])
        scores = self.np.tanh(X @ self.Wa) @ self.v
        weights = softmax(self.np, scores, axis=1)
        context = self.np.einsum("st,stf->sf", weights, X)
        self.context_train = context
        self.y_train = y
        self.weights = weights
        return self

    def predict(self, X):
        scores = self.np.tanh(X @ self.Wa) @ self.v
        weights = softmax(self.np, scores, axis=1)
        context = self.np.einsum("st,stf->sf", weights, X)
        return ridge_predict(self.np, self.context_train, self.y_train, context, self.alpha)


def train_models(config):
    np = load_dependencies()

    if config["data_source"] == "Fichiers .npy":
        X_train, y_train, X_test, y_test = load_npy_dataset(np, config["data_dir"])
    else:
        X_train, y_train, X_test, y_test = split_synthetic_data(
            np,
            config["n_samples"],
            config["seq_len"],
            config["n_features"],
            config["test_size"],
        )

    X_train_flat = flatten_sequences(X_train)
    X_test_flat = flatten_sequences(X_test)

    ridge_preds = ridge_predict(
        np,
        X_train_flat,
        y_train,
        X_test_flat,
        config["ridge_alpha"],
    )

    mlp_rnn_preds = random_feature_predict(
        np,
        X_train_flat,
        y_train,
        X_test_flat,
        config["mlp_hidden_size"],
        config["ridge_alpha"],
        activation="tanh",
    )

    mlp_lstm_preds = random_feature_predict(
        np,
        X_train_flat,
        y_train,
        X_test_flat,
        tuple(x * 2 for x in config["mlp_hidden_size"]),
        config["ridge_alpha"],
        activation="relu",
    )

    attn_model = SimpleAttentionModel(
        np,
        attention_dim=config["attention_dim"],
        alpha=config["ridge_alpha"],
    )
    attn_model.fit(X_train, y_train)
    attn_preds = attn_model.predict(X_test)

    predictions = {
        "Ridge": ridge_preds,
        "MLP-RNN": mlp_rnn_preds,
        "MLP-LSTM": mlp_lstm_preds,
        "RNN+Attention": attn_preds,
    }
    metrics = {
        name: calculate_metrics(np, preds, y_test)
        for name, preds in predictions.items()
    }

    return {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
    }


def render_sidebar():
    with st.sidebar:
        st.markdown("### Configuration")

        data_source = st.radio(
            "Source des donnees",
            ["Synthetiques", "Fichiers .npy"],
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
            n_samples, seq_len, n_features, test_size = 300, 40, 15, 20
            st.caption("Attendus : X_train.npy, y_train.npy, X_test.npy, y_test.npy")

        st.subheader("Hyperparametres")
        ridge_alpha = st.slider("Ridge alpha", 0.1, 10.0, 1.0)
        mlp_hidden_size = st.selectbox(
            "Couches cachees MLP",
            [(64, 32), (128, 64), (256, 128, 64)],
        )
        attention_dim = st.slider("Attention dimension", 8, 64, 32, step=8)
        run_button = st.button("Executer l'analyse", use_container_width=True)
        clear_cache_button = st.button("Vider le cache", use_container_width=True)

        if clear_cache_button:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("results", None)
            st.success("Cache Streamlit vide.")

    return {
        "data_source": data_source,
        "data_dir": data_dir,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "n_features": n_features,
        "test_size": test_size,
        "ridge_alpha": ridge_alpha,
        "mlp_hidden_size": mlp_hidden_size,
        "attention_dim": attention_dim,
        "run_button": run_button,
        "clear_cache_button": clear_cache_button,
    }


def render_results(results):
    metrics = results["metrics"]
    y_test = results["y_test"]
    predictions = results["predictions"]
    model_names = list(metrics.keys())

    st.markdown("<div class='sub-header'>Resultats</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    for col, name in zip(cols, model_names):
        with col:
            st.markdown(f"### {name}")
            st.metric("MSE", f"{metrics[name]['MSE']:.6f}")
            st.metric("RMSE", f"{metrics[name]['RMSE']:.6f}")
            st.metric("MAE", f"{metrics[name]['MAE']:.6f}")

    rows = []
    for name in model_names:
        rows.append(
            {
                "Modele": name,
                "MSE": round(metrics[name]["MSE"], 6),
                "RMSE": round(metrics[name]["RMSE"], 6),
                "MAE": round(metrics[name]["MAE"], 6),
            }
        )
    st.table(rows)

    best_model = min(model_names, key=lambda name: metrics[name]["RMSE"])
    st.success(
        f"Meilleur modele : {best_model} "
        f"(RMSE = {metrics[best_model]['RMSE']:.6f})"
    )

    n_plot = min(100, len(y_test))
    chart_data = {"Reel": y_test[:n_plot].tolist()}
    for name in model_names:
        chart_data[name] = predictions[name][:n_plot].tolist()

    st.markdown("### Predictions vs valeurs reelles")
    st.line_chart(chart_data)

    st.markdown("### Guide des modeles")
    st.markdown(
        """
        **Ridge** : modele lineaire rapide.

        **MLP-RNN** : reseau multicouche avec activation tanh.

        **MLP-LSTM** : reseau plus profond avec activation relu.

        **RNN+Attention** : moyenne ponderee des pas de temps avec attention.
        """
    )


def main():
    add_style()
    st.markdown("<div class='main-header'>RNN/LSTM Interface</div>", unsafe_allow_html=True)
    st.markdown("Comparaison de modeles pour la prediction de series financieres")
    st.markdown("---")

    config = render_sidebar()

    if config["run_button"]:
        try:
            with st.spinner("Execution de l'analyse..."):
                st.session_state["results"] = train_models(config)
        except Exception as exc:
            st.error("Erreur pendant l'execution de l'application.")
            st.exception(exc)
            with st.expander("Traceback technique"):
                st.code(traceback.format_exc())
            return

    if "results" not in st.session_state:
        st.info(
            "Configurez les parametres dans la barre laterale, puis cliquez sur "
            "\"Executer l'analyse\"."
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
        return

    render_results(st.session_state["results"])


try:
    main()
except Exception as exc:
    st.error("Erreur au demarrage de l'application.")
    st.exception(exc)
    with st.expander("Traceback technique"):
        st.code(traceback.format_exc())
