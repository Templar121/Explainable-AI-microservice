import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Disable internal GUI triggers
Image._showxv = lambda *args, **kwargs: None
Image.show = lambda *args, **kwargs: None

# --- Label Mapping ---
label_mapping = {
    0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc',
    4: 'akiec', 5: 'vasc', 6: 'df'
}
class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]

# --- Load Keras Model Once ---
ROUTES_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROUTES_DIR / "model" / "best_model.keras"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH!r}")

model = load_model(str(MODEL_PATH), compile=False)
print(f"[INFO] Model loaded from {MODEL_PATH}")

# --- Helpers ---
def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 64, 64, 3)

def preprocess_image_from_array(img_array):
    img = tf.image.resize(img_array, (64, 64)).numpy().astype(np.float32)
    return np.expand_dims(img, axis=0)

def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# --- SHAP Explanation ---
def explain_with_shap(image_path: str) -> str | None:
    try:
        img = Image.open(image_path).convert('RGB').resize((64, 64))
        arr = np.asarray(img, dtype=np.float32)
        preds = model.predict(preprocess_image(image_path), verbose=0)[0]

        masker = shap.maskers.Image("inpaint_telea", arr.shape)
        def shap_predict(x):
            batch = np.stack([preprocess_image_from_array(im)[0] for im in x])
            return model.predict(batch, verbose=0)

        explainer = shap.Explainer(shap_predict, masker, output_names=class_names)
        explanation = explainer(arr[np.newaxis, ...], max_evals=50)
        shap_vals = explanation.values[0]

        top_cls = np.argsort(preds)[::-1][:2]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, cls in zip(axes, top_cls):
            overlay = shap_vals[:, :, :, cls].mean(axis=-1)
            overlay -= overlay.mean()
            overlay /= (np.abs(overlay).max() + 1e-8)
            ax.imshow(arr.astype(np.uint8))
            ax.imshow(overlay, cmap='bwr', alpha=0.5, vmin=-1, vmax=1)
            ax.set_title(f"{class_names[cls]} ({preds[cls]:.2f})")
            ax.axis('off')
        return fig_to_base64(fig)
    except Exception as e:
        print(f"[ERROR] SHAP explanation failed: {e}")
        return None

# --- Occlusion Explanation ---
def explain_with_occlusion(image_path: str) -> str | None:
    try:
        img = Image.open(image_path).convert('RGB').resize((64, 64))
        arr = np.asarray(img, dtype=np.float32)
        input_tensor = preprocess_image_from_array(arr)
        preds = model.predict(input_tensor, verbose=0)[0]
        class_index = int(np.argmax(preds))
        confidence = preds[class_index]

        patch_size = 15
        stride = 8
        mean_pixel = np.mean(arr, axis=(0, 1), keepdims=True)
        heatmap = np.zeros((64, 64), dtype=np.float32)
        for y in range(0, 64 - patch_size + 1, stride):
            for x in range(0, 64 - patch_size + 1, stride):
                occluded = arr.copy()
                occluded[y:y+patch_size, x:x+patch_size] = mean_pixel
                pred = model.predict(preprocess_image_from_array(occluded), verbose=0)[0]
                drop = confidence - pred[class_index]
                heatmap[y:y+patch_size, x:x+patch_size] = drop

        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(arr.astype(np.uint8), alpha=0.6)
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.set_title(f"Occlusion Map ({class_names[class_index]})")
        ax.axis('off')
        return fig_to_base64(fig)
    except Exception as e:
        print(f"[ERROR] Occlusion explanation failed: {e}")
        return None

# --- Unified Entrypoint ---
def explain_image(image_path: str) -> dict:
    shap_res = explain_with_shap(image_path)
    occl_res = explain_with_occlusion(image_path)

    print("[DEBUG] SHAP length:", len(shap_res) if shap_res else None)
    print("[DEBUG] OCCL length:", len(occl_res) if occl_res else None)

    return {
        "shap_base64": shap_res,
        "occlusion_base64": occl_res
    }

