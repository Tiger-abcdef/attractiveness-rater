import io
import os

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# Must be the first Streamlit command
st.set_page_config(page_title="Attractiveness Rater", layout="centered")

st.write("RUNNING FILE:", os.path.abspath(__file__))

# -------------------------
# Model (must match training)
# -------------------------
class BeautyRegressor(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats).squeeze(1)
        return out

IMG_SIZE = 224
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

@st.cache_resource
def load_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeautyRegressor("efficientnet_b0", dropout=0.2).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

def predict_score(model, device, pil_img: Image.Image) -> float:
    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).item()
    return float(pred)

def clamp_score(score: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, score))

# -------------------------
# UI
# -------------------------
st.title("Attractiveness Rater")

WEIGHTS_PATH = "best_beauty_regressor.pth"
model, device = load_model(WEIGHTS_PATH)

uploaded = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
)

if uploaded is not None:
    st.session_state["uploaded_bytes"] = uploaded.getvalue()

if "uploaded_bytes" in st.session_state:
    if st.button("Test"):
        img_bytes = st.session_state["uploaded_bytes"]

        try:
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert("RGB")  # standardize
            img.load()                # force full decode now
        except Exception as e:
            st.error(f"Could not decode this image. Try a different image.\n\nError: {e}")
            st.stop()

        score_raw = predict_score(model, device, img)
        score = clamp_score(score_raw, 1.0, 5.0)

        st.image(img, use_column_width=True)
        st.subheader(f"Score: {score:.2f} / 5.00")

        percent = (score - 1.0) / 4.0
        st.progress(int(percent * 100))

        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; width:100%; font-size:14px;">
              <span>1</span><span>2</span><span>3</span><span>4</span><span>5</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

