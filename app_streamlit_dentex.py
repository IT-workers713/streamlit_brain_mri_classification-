import io
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from skimage.transform import resize

st.set_page_config(
    page_title="Classification dentaire - Dentex",
    page_icon="🦷",
    layout="wide",
)

MODEL_DEFAULT_PATH = "best_texture_classifier.pkl"

# ============================================================
# CLASSES FINALES DE CLASSIFICATION (category_id_3)
# ============================================================

CLASS_NAMES = {
    0: "Impacted",
    1: "Caries",
    2: "Deep Caries",
    3: "Periapical Lesion",
}

CLASS_NAMES_LIST = [
    "Impacted",
    "Caries",
    "Deep Caries",
    "Periapical Lesion",
]

# ============================================================
# 1) FONCTIONS DE PRÉTRAITEMENT (ALIGNÉES AVEC L'ENTRAÎNEMENT)
# ============================================================

def load_uploaded_image(uploaded_file) -> np.ndarray:
    """Charge une image PNG/JPG en niveaux de gris uint8."""
    image = Image.open(uploaded_file).convert("L")
    return np.array(image, dtype=np.uint8)


@st.cache_resource(show_spinner=False)
def load_model_artifacts(model_path: str):
    return joblib.load(model_path)


def get_texture_features(roi: np.ndarray) -> np.ndarray:
    """Extraction des caractéristiques de texture : CLAHE + LBP + Gabor."""
    roi_res = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)

    if roi_res.dtype != np.uint8:
        roi_res = np.clip(roi_res, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_norm = clahe.apply(roi_res)

    # LBP
    lbp = local_binary_pattern(roi_norm, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-7)

    # Gabor
    g_feats = []
    for theta in [0, np.pi / 4, np.pi / 2]:
        kernel = np.real(gabor_kernel(0.1, theta=theta))
        f = ndi.convolve(roi_norm, kernel, mode="wrap")
        g_feats.extend([float(f.mean()), float(f.std())])

    return np.concatenate([hist, np.array(g_feats, dtype=np.float32)])


def extract_image_features(images: np.ndarray) -> np.ndarray:
    """
    Support si le pipeline sauvegardé n'utilise pas use_flattening=True.
    images: (N, H, W, C) ou (N, H, W)
    """
    features_list = []

    for i in range(images.shape[0]):
        img = images[i]

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        mean_val = float(np.mean(img))
        std_val = float(np.std(img))

        hist_range = (0, 1) if np.max(img) <= 1.0 else (0, 255)
        hist = np.histogram(img.flatten(), bins=32, range=hist_range)[0]

        if img.ndim == 2:
            lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
            lbp_hist = np.histogram(lbp.flatten(), bins=10)[0]
            features = [mean_val, std_val] + hist.tolist() + lbp_hist.tolist()
        else:
            img_gray = np.mean(img, axis=2)
            lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
            lbp_hist = np.histogram(lbp.flatten(), bins=10)[0]
            channel_means = np.mean(img, axis=(0, 1))
            channel_stds = np.std(img, axis=(0, 1))
            features = (
                [mean_val, std_val]
                + channel_means.tolist()
                + channel_stds.tolist()
                + hist.tolist()
                + lbp_hist.tolist()
            )

        features_list.append(features)

    return np.array(features_list, dtype=np.float32)


def prepare_image_branch(gray_image: np.ndarray, artifacts: Dict) -> np.ndarray:
    """Prépare la branche image selon le pipeline sauvegardé."""
    use_flattening = artifacts.get("use_flattening", True)

    img_norm = gray_image.astype(np.float32) / 255.0

    if use_flattening:
        target_size = tuple(artifacts.get("target_size", (32, 32)))
        img_resized = resize(img_norm, target_size, anti_aliasing=True)
        X_img = np.array([img_resized.flatten()], dtype=np.float32)

        scaler_img = artifacts.get("scaler_img")
        if scaler_img is not None:
            X_img = scaler_img.transform(X_img)
    else:
        img_resized = resize(img_norm, (64, 64), anti_aliasing=True)
        X_img_4d = np.array([img_resized], dtype=np.float32).reshape(1, 64, 64, 1)
        X_img = extract_image_features(X_img_4d)

        scaler_img = artifacts.get("scaler_img")
        if scaler_img is not None:
            X_img = scaler_img.transform(X_img)

    pca_img = artifacts.get("pca_img")
    if pca_img is not None:
        X_img = pca_img.transform(X_img)

    return X_img


def prepare_texture_branch(gray_image: np.ndarray, artifacts: Dict) -> np.ndarray:
    """Prépare la branche texture selon le pipeline sauvegardé."""
    tex_features = get_texture_features(gray_image).reshape(1, -1)

    scaler_tex = artifacts.get("scaler_tex")
    if scaler_tex is not None:
        tex_features = scaler_tex.transform(tex_features)

    pca_tex = artifacts.get("pca_tex")
    if pca_tex is not None:
        tex_features = pca_tex.transform(tex_features)

    return tex_features


def predict_single_image(gray_image: np.ndarray, artifacts: Dict) -> Tuple[str, float, pd.DataFrame]:
    """Prédit une classe sur une image/crop grayscale."""
    X_img = prepare_image_branch(gray_image, artifacts)
    X_tex = prepare_texture_branch(gray_image, artifacts)

    X_combined = np.hstack((X_img, X_tex))

    selector = artifacts.get("selector")
    if selector is not None:
        X_combined = selector.transform(X_combined)

    model = artifacts["model"]
    prediction = model.predict(X_combined)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_combined)[0]
    else:
        probabilities = np.zeros(len(CLASS_NAMES_LIST), dtype=np.float32)
        probabilities[int(prediction)] = 1.0

    label_encoder = artifacts.get("label_encoder")
    if label_encoder is not None:
        pred_label = label_encoder.inverse_transform([prediction])[0]
        classes = label_encoder.inverse_transform(np.arange(len(probabilities)))
    else:
        classes = np.array(
            artifacts.get("classes", CLASS_NAMES_LIST)
        )
        pred_label = classes[int(prediction)] if len(classes) > int(prediction) else str(prediction)

    df_probs = pd.DataFrame({
        "Classe": classes,
        "Probabilité": probabilities,
    }).sort_values("Probabilité", ascending=False)

    confidence = float(df_probs.iloc[0]["Probabilité"])
    return str(pred_label), confidence, df_probs


def plot_probabilities(df_probs: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df_probs["Classe"], df_probs["Probabilité"])
    ax.set_title("Probabilités par classe")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Probabilité")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig


def apply_manual_crop(gray_image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    h_img, w_img = gray_image.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    return gray_image[y:y + h, x:x + w]


# ============================================================
# 2) UI STREAMLIT
# ============================================================

st.title("🦷 Application Streamlit de classification dentaire")
st.caption(
    "Classification à partir d'images PNG/JPG avec un pipeline hybride : "
    "image + texture + modèle ML sauvegardé dans model.pkl"
)

st.markdown("""
### Classification finale
Le modèle effectue une **classification multiclasse des anomalies dentaires**
selon la variable cible **`category_id_3`** avec les 4 classes finales suivantes :

- **Impacted**
- **Caries**
- **Deep Caries**
- **Periapical Lesion**
""")

with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Chemin du modèle .pkl", value=MODEL_DEFAULT_PATH)
    st.markdown(
        "**Cible finale :** `category_id_3`  \n"
        "**Classes :** Impacted, Caries, Deep Caries, Periapical Lesion  \n"
        "**Formats acceptés :** PNG, JPG, JPEG  \n"
        "**Conseil :** le modèle a été entraîné sur des **ROI/crops dentaires**, "
        "pas sur des radiographies panoramiques complètes."
    )

artifacts = None
model_loaded = False

try:
    artifacts = load_model_artifacts(model_path)
    model_loaded = True
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.info(
        "Place ton fichier `best_texture_classifier.pkl` dans le même dossier que ce script, "
        "ou indique son chemin exact dans la barre latérale."
    )

if model_loaded:
    classes = artifacts.get("classes")
    best_score = artifacts.get("best_score")

    c1, c2 = st.columns(2)
    with c1:
        st.success("Modèle chargé avec succès.")
    with c2:
        if best_score is not None:
            st.info(f"Score sauvegardé : {best_score:.4f}")

    if classes is not None:
        st.write("**Classes connues par le modèle :**", ", ".join([str(c) for c in classes]))
    else:
        st.write("**Classes finales :**", ", ".join(CLASS_NAMES_LIST))

uploaded_files = st.file_uploader(
    "Importer une ou plusieurs images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files and model_loaded:
    st.divider()

    use_crop = False
    crop_params = None

    if len(uploaded_files) == 1:
        preview_image = load_uploaded_image(uploaded_files[0])
        h_img, w_img = preview_image.shape

        with st.sidebar:
            st.subheader("Recadrage manuel (optionnel)")
            use_crop = st.checkbox("Activer le recadrage ROI", value=False)
            if use_crop:
                x = st.slider("x", 0, max(0, w_img - 1), 0)
                y = st.slider("y", 0, max(0, h_img - 1), 0)
                w = st.slider("largeur", 1, max(1, w_img - x), max(1, w_img - x))
                h = st.slider("hauteur", 1, max(1, h_img - y), max(1, h_img - y))
                crop_params = (x, y, w, h)

    results: List[Dict] = []

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        uploaded_file.seek(0)
        gray_image = load_uploaded_image(uploaded_file)
        original_gray = gray_image.copy()

        if len(uploaded_files) == 1 and use_crop and crop_params is not None:
            gray_image = apply_manual_crop(gray_image, *crop_params)
            if gray_image.size == 0:
                st.warning("Le crop est vide. Vérifie les coordonnées du recadrage.")
                continue

        pred_label, confidence, df_probs = predict_single_image(gray_image, artifacts)

        results.append({
            "fichier": uploaded_file.name,
            "classification_finale": pred_label,
            "confiance": confidence,
        })

        with st.container():
            st.subheader(f"Image {idx} — {uploaded_file.name}")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("**Image analysée**")
                st.image(original_gray, caption="Image originale", use_container_width=True, clamp=True)

                if len(uploaded_files) == 1 and use_crop and crop_params is not None:
                    st.image(gray_image, caption="ROI recadrée", use_container_width=True, clamp=True)

            with col2:
                st.metric("Classe prédite", pred_label)
                st.metric("Confiance", f"{confidence:.2%}")

                st.success(f"Classification finale : {pred_label}")
                st.write(
                    f"Le modèle a classé cette image dans la catégorie finale "
                    f"**{pred_label}** avec une confiance de **{confidence:.2%}**."
                )

                st.write("**Type de tâche :** classification multiclasse sur `category_id_3`")
                st.write("**Classes possibles :** Impacted, Caries, Deep Caries, Periapical Lesion")

                st.dataframe(df_probs, use_container_width=True, hide_index=True)
                st.pyplot(plot_probabilities(df_probs))

            st.divider()

    if results:
        df_results = pd.DataFrame(results)
        st.subheader("Résumé des classifications finales")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger les résultats CSV",
            data=csv,
            file_name="predictions_dentex.csv",
            mime="text/csv",
        )

with st.expander("Aperçu des annotations et définition de la classification"):
    st.markdown("""
**Aperçu des données :**
- `images` contient les métadonnées des radiographies.
- `annotations` contient les boîtes englobantes (`bbox`) et les labels associés.
- `categories_1` et `categories_2` représentent d'autres niveaux d'annotation.
- **`categories_3` représente la classification finale utilisée par le modèle.**

**Classification finale utilisée :**
- `category_id_3 = 0` → **Impacted**
- `category_id_3 = 1` → **Caries**
- `category_id_3 = 2` → **Deep Caries**
- `category_id_3 = 3` → **Periapical Lesion**

Ainsi, le modèle effectue une **classification finale multiclasse des anomalies dentaires**
sur 4 catégories.
""")

with st.expander("Notes importantes"):
    st.markdown("""
- Ce pipeline reproduit ton approche **hybride** :
  - branche **image**,
  - branche **texture**,
  - combinaison des features,
  - sélection de caractéristiques,
  - classification finale via le modèle sauvegardé.
- Le modèle a été entraîné sur des **boîtes englobantes (bbox)** issues des annotations Dentex.
- Donc, pour une **radio panoramique complète**, il vaut mieux :
  1. recadrer manuellement une dent / lésion,
  2. ou ajouter ensuite un module de détection avant la classification.
- Si ton `model.pkl` contient `predict_proba`, l'application affichera automatiquement les probabilités.
""")