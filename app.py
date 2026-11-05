import io, numpy as np, torch, torch.nn as nn, streamlit as st
from PIL import Image
from torchvision import models, transforms
from matplotlib import cm

st.set_page_config(page_title="Illegal Mining Detection", page_icon="⛏️", layout="wide")

@st.cache_resource
def load():
    ck = torch.load("Models/mining_effnet.pt", map_location="cpu")
    m = models.efficientnet_b0()
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck

model, ck = load()
MI, IMG = ck["mining_idx"], ck["img_size"]
tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def analyze(img):
    x = tf(img).unsqueeze(0)
    with torch.enable_grad():
        f = model.features(x)
        logits = model.classifier(torch.flatten(model.avgpool(f), 1))
        g = torch.autograd.grad(logits[0, MI], f)[0]
    prob = logits.softmax(1)[0, MI].item()
    cam = torch.relu((g.mean((2, 3), keepdim=True) * f).sum(1))[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return prob, cam

def overlay(img, cam, alpha=0.45):
    c = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)
    heat = Image.fromarray((cm.jet(np.array(c) / 255.0)[..., :3] * 255).astype(np.uint8))
    return Image.blend(img.convert("RGB"), heat, alpha)

st.title("⛏️ Illegal Mining Detection from Aerial Images")
st.caption("EfficientNet-B0 transfer-learning classifier with Grad-CAM explanations")

with st.sidebar:
    thr = st.slider("Decision threshold", 0.05, 0.95, float(round(ck["threshold"], 2)), 0.01,
                    help="Raise for higher precision (fewer false alarms), lower for higher recall.")
    show_cam = st.checkbox("Show Grad-CAM heatmap", True)

files = st.file_uploader("Upload aerial image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
rows = []
for f in files or []:
    img = Image.open(io.BytesIO(f.read())).convert("RGB")
    prob, cam = analyze(img)
    is_mine = prob >= thr
    rows.append({"File": f.name, "Mining probability": round(prob, 4),
                 "Prediction": "Illegal mining" if is_mine else "No mining"})
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.image(img, caption=f.name, use_container_width=True)
    if show_cam:
        c2.image(overlay(img, cam), caption="Grad-CAM (regions driving the decision)", use_container_width=True)
    (c3.error if is_mine else c3.success)("⚠️ Illegal mining detected" if is_mine else "✅ No mining activity")
    c3.metric("Mining probability", f"{prob*100:.1f}%")
    c3.progress(min(prob, 1.0))
    st.divider()

if len(rows) > 1:
    st.subheader("Batch summary")
    st.dataframe(rows, use_container_width=True)
    st.download_button("Download CSV", "\n".join(
        [",".join(rows[0].keys())] + [",".join(str(v) for v in r.values()) for r in rows]),
        "results.csv", "text/csv")
