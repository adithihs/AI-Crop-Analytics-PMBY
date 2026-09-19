"""
Crop Health Analysis — specimen sheet interface
Streamlit front end for the hybrid ResNet50 + DenseNet121 crop classifier.

Design: the page is a mounted specimen sheet. Results print as a
determination label — ruled fields, dotted leaders, botanical ink.

File layout
-----------
1. Imports & paths
2. Domain constants
3. Theme (CSS)
4. Backend: model loading, preprocessing, inference   <- unchanged logic
5. UI components
6. main()
"""

# ════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS & PATHS
# ════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from models.hybrid_model import load_checkpoint  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT_DIR / "models" / "hybrid_resnet_densenet_checkpoint.pth"

SAMPLE_IMAGE_URL = (
    "https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=600&q=80"
)


# ════════════════════════════════════════════════════════════════════════════
# 2. DOMAIN CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

PLANT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

CROP_SCIENTIFIC = {
    "Apple": "Malus domestica",
    "Blueberry": "Vaccinium sect. Cyanococcus",
    "Cherry": "Prunus avium",
    "Corn": "Zea mays",
    "Grape": "Vitis vinifera",
    "Orange": "Citrus sinensis",
    "Peach": "Prunus persica",
    "Pepper": "Capsicum annuum",
    "Potato": "Solanum tuberosum",
    "Raspberry": "Rubus idaeus",
    "Soybean": "Glycine max",
    "Squash": "Cucurbita pepo",
    "Strawberry": "Fragaria ananassa",
    "Tomato": "Solanum lycopersicum",
}

CROP_SEASONS = {
    "Corn": "Kharif (June – October)",
    "Potato": "Rabi (October – March)",
    "Tomato": "Rabi / Year-round",
    "Soybean": "Kharif (June – September)",
    "Apple": "Rabi / Zaid",
    "Grape": "Annual",
    "Strawberry": "Rabi (October – March)",
    "Pepper": "Kharif (July – November)",
    "Peach": "Rabi (March – June)",
    "Orange": "Annual",
    "Blueberry": "Zaid (March – June)",
    "Cherry": "Zaid (April – June)",
    "Raspberry": "Zaid / Annual",
    "Squash": "Kharif / Rabi",
}

GROWTH_STAGES = [
    "Germination / Seedling",
    "Vegetative Growth",
    "Tillering / Branching",
    "Stem Extension",
    "Heading / Flowering",
    "Grain Filling / Fruiting",
    "Maturity / Harvest Ready",
]

GROWTH_ADVICE = {
    "Germination / Seedling": "Ensure adequate moisture. Avoid waterlogging. Thin seedlings if overcrowded.",
    "Vegetative Growth": "Apply nitrogen-rich fertiliser. Monitor for early pest infestation.",
    "Tillering / Branching": "Irrigation is critical at this stage. Watch for stem borer and aphids.",
    "Stem Extension": "Apply potassium fertiliser. Provide support for tall varieties.",
    "Heading / Flowering": "Avoid pesticide spray during flowering. Critical water requirement period.",
    "Grain Filling / Fruiting": "Ensure consistent moisture. Monitor for fungal diseases and fruit borer.",
    "Maturity / Harvest Ready": "Reduce irrigation gradually. Prepare harvesting equipment. Monitor for storage pests.",
}

LOW_CONF_THRESHOLD = 35

UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]


# ════════════════════════════════════════════════════════════════════════════
# 3. THEME
# ════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@400;500;600&display=swap');

:root {
    --desk:      #dfe4da;
    --paper:     #fbfaf6;
    --ink:       #16241b;
    --ink-soft:  #66766b;
    --rule:      #d8ded3;
    --rule-soft: #e9ede5;
    --leaf:      #2f6b43;
    --ochre:     #a8701f;
    --rust:      #9c3b2c;
}

/* ---------- desk ---------- */
.stApp { background: var(--desk); }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* ---------- the sheet ---------- */
.block-container {
    max-width: 940px;
    background: var(--paper);
    padding: 3.2rem 3.4rem 3rem !important;
    margin: 2.2rem auto 3rem;
    border: 1px solid #c9d1c5;
    box-shadow: 0 1px 0 #fff inset, 0 10px 30px rgba(22,36,27,0.09);
}

/* ---------- masthead ---------- */
.sheet-no {
    font-size: 0.7rem; color: var(--ink-soft); letter-spacing: 0.06em;
    border-bottom: 1px solid var(--rule); padding-bottom: 0.7rem;
    display: flex; justify-content: space-between;
}
.masthead { padding: 2.1rem 0 1.9rem; border-bottom: 3px double var(--rule); }
.masthead h1 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: clamp(2rem, 4.2vw, 2.9rem); font-weight: 500;
    line-height: 1.08; letter-spacing: -0.015em; margin: 0 0 0.9rem;
    color: var(--ink);
}
.masthead p {
    font-family: 'Newsreader', Georgia, serif; font-size: 1.02rem;
    color: var(--ink-soft); line-height: 1.7; max-width: 60ch; margin: 0;
}

/* ---------- field headings ---------- */
.field-head {
    font-family: 'Newsreader', Georgia, serif; font-style: italic;
    font-size: 1.12rem; color: var(--ink); margin: 2.4rem 0 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--rule);
}

/* ---------- mount window ---------- */
.mount-note {
    font-family: 'Newsreader', Georgia, serif; font-size: 0.96rem;
    color: var(--ink-soft); line-height: 1.7; margin: 0.9rem 0 0;
}
.caption { font-size: 0.74rem; color: var(--ink-soft); margin-top: 0.55rem; letter-spacing: 0.02em; }

/* ---------- determination label ---------- */
.label-card {
    background: #fff; border: 1px solid var(--rule);
    box-shadow: 0 1px 2px rgba(22,36,27,0.05);
    padding: 1.15rem 1.25rem 1.25rem; height: 100%;
}
.label-card h4 {
    font-family: 'Inter', sans-serif; font-size: 0.73rem; font-weight: 600;
    color: var(--leaf); letter-spacing: 0.04em; margin: 0 0 0.9rem;
    padding-bottom: 0.55rem; border-bottom: 1px solid var(--rule-soft);
}
.fld { display: flex; align-items: baseline; gap: 0.4rem; padding: 0.42rem 0; font-size: 0.86rem; }
.fld .k { color: var(--ink-soft); white-space: nowrap; font-size: 0.8rem; }
.fld .dots { flex: 1; border-bottom: 1px dotted #c9d2c6; transform: translateY(-3px); }
.fld .v { font-weight: 500; text-align: right; white-space: nowrap; }
.fld .v.leaf { color: var(--leaf); }
.fld .v.rust { color: var(--rust); }
.fld .v.ochre { color: var(--ochre); }
.fld .v.sci {
    font-family: 'Newsreader', Georgia, serif; font-style: italic;
    font-weight: 400; font-size: 0.93rem; white-space: normal;
}

/* ---------- colour safety net ----------
   The host page can inject its own text colour on generic <p>/<span>
   elements. Every text node below is given an explicit colour so
   nothing is left to inheritance. */
.block-container, .block-container * { color: var(--ink) !important; }
.field-head, .masthead h1 { color: var(--ink) !important; }
.masthead p, .mount-note, .prose { color: var(--ink) !important; }
.sheet-no, .caption, .scale-cap, .colophon,
.fld .k, [data-testid="stFileUploader"] small { color: var(--ink-soft) !important; }
.fld .v.leaf, .scale-cap.leaf { color: var(--leaf) !important; }
.fld .v.rust { color: var(--rust) !important; }
.fld .v.ochre { color: var(--ochre) !important; }
.slip.warn .prose { color: var(--rust) !important; }
.label-card h4 { color: var(--leaf) !important; }
.stButton > button { color: #fbfaf6 !important; }
.stButton > button:disabled { color: #a3ada5 !important; }

/* ---------- growth scale ---------- */
.scale { display: flex; align-items: flex-end; gap: 3px; height: 30px; margin-top: 0.7rem; }
.tick { flex: 1; background: var(--rule); height: 9px; }
.tick.done { background: #9dbfa8; }
.tick.now  { background: var(--leaf); height: 22px; }
.scale-cap { font-size: 0.76rem; color: var(--ink-soft); margin-top: 0.45rem; }

/* ---------- prose / notices ---------- */
.prose {
    font-family: 'Newsreader', Georgia, serif; font-size: 1rem;
    line-height: 1.72; color: var(--ink); margin: 0;
}
.slip {
    background: #fff; border: 1px solid var(--rule);
    border-left: 3px solid var(--leaf);
    padding: 1rem 1.2rem;
}
.slip.warn { border-left-color: var(--rust); }
.slip.warn .prose { color: var(--rust); }

/* ---------- widgets ---------- */
.stButton > button {
    background: var(--leaf) !important; color: #fbfaf6 !important;
    border: 1px solid #245536 !important; border-radius: 2px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    font-size: 0.87rem !important; letter-spacing: 0.03em !important;
    padding: 0.62rem 1.2rem !important; width: 100% !important;
    transition: background 0.15s !important;
}
.stButton > button:hover:enabled { background: #245536 !important; }
.stButton > button:disabled {
    background: transparent !important; color: #a3ada5 !important;
    border: 1px dashed var(--rule) !important;
}
.stButton > button:focus-visible { outline: 2px solid var(--ochre) !important; outline-offset: 2px !important; }

[data-testid="stFileUploader"] {
    background: #fff; border: 1px solid var(--rule); border-radius: 0;
    padding: 0.5rem 0.9rem;
}
[data-testid="stFileUploader"] section { background: transparent; border: none; padding: 0.4rem 0; }
[data-testid="stFileUploader"] small { color: var(--ink-soft); }

[data-testid="stImage"] img { border: 1px solid var(--rule); background: #fff; padding: 6px; }

/* ---------- colophon ---------- */
.colophon {
    border-top: 3px double var(--rule); margin-top: 3rem; padding-top: 1.2rem;
    font-size: 0.76rem; color: var(--ink-soft); line-height: 1.75;
    display: flex; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap;
}

@media (max-width: 640px) {
    .block-container { padding: 2rem 1.4rem !important; margin: 0.8rem auto; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""


# ════════════════════════════════════════════════════════════════════════════
# 4. BACKEND — unchanged
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_crop_model():
    """Load the internship hybrid ResNet50 + DenseNet121 model."""
    if not MODEL_PATH.exists():
        return None, None, f"Model checkpoint not found: {MODEL_PATH}"

    try:
        model, class_names = load_checkpoint(str(MODEL_PATH), DEVICE)
        return model, class_names, None
    except Exception as exc:
        return None, None, f"Could not load hybrid model: {exc}"


def preprocess_image(image: Image.Image):
    """Prepare an image using the same ImageNet normalization used in training."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def infer(image: Image.Image, model, class_names) -> dict:
    """Run hybrid model inference and build the application result payload."""
    tensor = preprocess_image(image)

    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    conf_pct = round(confidence * 100, 1)

    k = min(5, len(class_names))
    top_probs, top_indices = torch.topk(probs, k=k)
    top5 = [
        (class_names[int(idx)], round(float(prob) * 100, 1))
        for prob, idx in zip(top_probs, top_indices)
    ]

    if conf_pct < LOW_CONF_THRESHOLD:
        return {
            "low_conf": True,
            "confidence": conf_pct,
            "top5": top5,
            "error_msg": (
                f"Model confidence is only {conf_pct}% — image may not show a "
                "supported crop leaf clearly."
            ),
        }

    predicted_class = class_names[pred_idx]
    parts = predicted_class.split("___")
    raw_crop = parts[0]
    raw_cond = parts[1] if len(parts) > 1 else "healthy"

    crop_name = raw_crop.split("(")[0].replace("_", " ").replace(",", "").strip()
    crop_key = crop_name.split()[0]

    disease_raw = raw_cond.replace("_", " ").strip()
    is_healthy = "healthy" in raw_cond.lower()
    health_str = "Healthy" if is_healthy else disease_raw

    sci_name = CROP_SCIENTIFIC.get(crop_key, "Data unavailable")
    season = CROP_SEASONS.get(crop_key, "Data unavailable")

    if is_healthy:
        severity = "None"
        affected_pct = "0%"
        insurance_flag = "Not Applicable"
        dmg_label = "None detected"
    else:
        if conf_pct >= 80:
            severity = "High"
            affected_pct = "60–80%"
        elif conf_pct >= 60:
            severity = "Moderate"
            affected_pct = "30–60%"
        else:
            severity = "Low"
            affected_pct = "10–30%"
        insurance_flag = "Eligible for Claim"
        dmg_label = disease_raw

    # Growth-stage estimate retained from the full team application.
    img_arr = np.array(image.resize((100, 100))).astype(float)
    green_ch = img_arr[:, :, 1].mean() / 255.0
    red_ch = img_arr[:, :, 0].mean() / 255.0

    vi = (green_ch - red_ch) / (green_ch + red_ch + 1e-6)

    if vi > 0.15:
        growth_idx = 5 if green_ch > 0.5 else 2
    elif vi > 0.05:
        growth_idx = 4
    elif red_ch > 0.55:
        growth_idx = 6
    elif green_ch < 0.3 and red_ch < 0.3:
        growth_idx = 0
    else:
        growth_idx = 3

    if not is_healthy and growth_idx > 4:
        growth_idx = 4

    growth_stage = GROWTH_STAGES[growth_idx]
    growth_advice = GROWTH_ADVICE[growth_stage]
    growth_pct = int((growth_idx + 1) / len(GROWTH_STAGES) * 100)

    return {
        "low_conf": False,
        "confidence": conf_pct,
        "top5": top5,
        "crop_name": crop_name,
        "scientific_name": sci_name,
        "season": season,
        "is_healthy": is_healthy,
        "health_status": health_str,
        "disease_name": dmg_label,
        "severity": severity,
        "affected_area": affected_pct,
        "insurance_flag": insurance_flag,
        "growth_stage": growth_stage,
        "growth_index": growth_idx,
        "growth_total": len(GROWTH_STAGES),
        "growth_pct": growth_pct,
        "growth_advice": growth_advice,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. UI
# ════════════════════════════════════════════════════════════════════════════

def setup_page() -> None:
    st.set_page_config(
        page_title="Crop Health Analysis",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)


def init_state() -> None:
    for key in ["uploaded_image", "analysis_result"]:
        if key not in st.session_state:
            st.session_state[key] = None


def field(label: str, value: str, tone: str = "") -> str:
    """One ruled field line with a dotted leader."""
    return (
        f'<div class="fld"><span class="k">{label}</span>'
        f'<span class="dots"></span>'
        f'<span class="v {tone}">{value}</span></div>'
    )


def render_masthead() -> None:
    st.markdown(
        """
        <div class="sheet-no">
            <span>AI Crop Analytics</span>
            <span>Leaf specimen examination</span>
        </div>
        <div class="masthead">
            <h1>AI for real-time crop analytics</h1>
            <p>Mount a single leaf in the frame below. The examination returns the
            crop and its botanical name, any disease showing on the tissue, how far
            the damage has spread, the stage of growth, and what the field needs next —
            all in a matter of seconds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mount(model) -> bool:
    st.markdown('<p class="field-head">The specimen</p>', unsafe_allow_html=True)

    form_col, mount_col = st.columns([1, 1], gap="large")

    with form_col:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=UPLOAD_TYPES,
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="mount-note">A good specimen fills the frame, sits in even '
            'light, and shows one leaf against a plain background. Blur and '
            'distance are what most often defeat the reading.</p>',
            unsafe_allow_html=True,
        )

    with mount_col:
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = image
            st.image(image, use_container_width=True)
            st.markdown(
                f'<p class="caption">{uploaded_file.name} · '
                f'{image.size[0]}×{image.size[1]} px</p>',
                unsafe_allow_html=True,
            )
        else:
            st.session_state.uploaded_image = None
            st.session_state.analysis_result = None
            st.image(SAMPLE_IMAGE_URL, use_container_width=True)
            st.markdown(
                '<p class="caption">Reference plate — mount your own to begin</p>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.3rem'></div>", unsafe_allow_html=True)

    if model is None:
        st.markdown(
            '<div class="slip warn"><p class="prose">The examination service is '
            'not responding. Try again in a few moments.</p></div>',
            unsafe_allow_html=True,
        )

    ready = st.session_state.uploaded_image is not None and model is not None

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        clicked = st.button(
            "Examine specimen" if ready else "Mount a specimen first",
            disabled=not ready,
            key="btn_analyse",
        )

    return clicked and ready


def _label_identification(r: dict) -> str:
    return (
        '<div class="label-card"><h4>Determination</h4>'
        + field("Crop", r["crop_name"], "leaf")
        + field("Botanical name", r["scientific_name"], "sci")
        + field("Season", r["season"])
        + "</div>"
    )


def _label_condition(r: dict) -> str:
    tone_health = "leaf" if r["is_healthy"] else "rust"
    tone_sev = "leaf" if r["is_healthy"] else ("rust" if r["severity"] == "High" else "ochre")
    tone_ins = "leaf" if r["is_healthy"] else "ochre"
    return (
        '<div class="label-card"><h4>Condition</h4>'
        + field("Status", r["health_status"], tone_health)
        + field("Lesion", r["disease_name"])
        + field("Severity", r["severity"], tone_sev)
        + field("Spread", r["affected_area"])
        + field("Insurance", r["insurance_flag"], tone_ins)
        + "</div>"
    )


def _label_growth(r: dict) -> str:
    ticks = ""
    for i in range(r["growth_total"]):
        cls = "now" if i == r["growth_index"] else ("done" if i < r["growth_index"] else "")
        ticks += f'<div class="tick {cls}"></div>'
    return (
        '<div class="label-card"><h4>Phenology</h4>'
        + field("Stage", r["growth_stage"], "leaf")
        + field("Position", f"{r['growth_index'] + 1} of {r['growth_total']}")
        + f'<div class="scale">{ticks}</div>'
        + f'<p class="scale-cap">{r["growth_pct"]}% through the season</p>'
        + "</div>"
    )


def render_label() -> None:
    st.markdown('<p class="field-head">The reading</p>', unsafe_allow_html=True)

    r = st.session_state.analysis_result

    if r is None:
        st.markdown(
            '<div class="slip"><p class="prose">Nothing mounted yet. Once a leaf '
            'is examined, its determination label prints here — name, condition, '
            'and the stage the crop has reached.</p></div>',
            unsafe_allow_html=True,
        )
        return

    if r.get("low_conf"):
        st.markdown(
            '<div class="slip warn"><p class="prose">This specimen cannot be '
            'determined. The tissue is unclear or the crop falls outside the '
            'reference collection. Mount a sharper, better-lit leaf and examine '
            'again.</p></div>',
            unsafe_allow_html=True,
        )
        return

    labels = [_label_identification(r), _label_condition(r), _label_growth(r)]
    for col, html in zip(st.columns(3, gap="medium"), labels):
        with col:
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="slip"><p class="prose"><b>Field note.</b> '
        f'{r["growth_advice"]}</p></div>',
        unsafe_allow_html=True,
    )


def render_colophon() -> None:
    st.markdown(
        """
        <div class="colophon">
            <span>Crop Health Analysis · leaf specimen examination</span>
            <span>A reading supports field inspection; it does not replace it.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    setup_page()
    init_state()
    render_masthead()

    model, model_classes, _ = load_crop_model()

    if render_mount(model):
        with st.spinner("Examining the specimen…"):
            st.session_state.analysis_result = infer(
                st.session_state.uploaded_image, model, model_classes
            )

    render_label()
    render_colophon()


if __name__ == "__main__":
    main()