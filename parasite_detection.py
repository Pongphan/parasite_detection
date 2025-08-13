import os
import io
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.losses import mean_squared_error

st.set_page_config(page_title="AI Detector", layout="wide")

default_model_dir = "model/"
model_dir = st.sidebar.text_input("Model directory", value=default_model_dir, help="Folder that contains the *.keras models")

CLASS_LABEL = ["Artifact", "As_fer", "As_unfer", "Hd", "Hn", "Hw", "Mif", "Ov", "Tn", "Tt"]

CLASS_CONFIGS = {
    1: ("cnn_asfer460.keras",  (460, 460)),
    2: ("cnn_asunfer650.keras",(650, 650)),
    3: ("cnn_hd480.keras",     (480, 480)),
    4: ("cnn_hn420.keras",     (420, 420)),
    5: ("cnn_hw460.keras",     (460, 460)),
    6: ("cnn_mif200.keras",    (200, 200)),
    7: ("cnn_ov200.keras",     (200, 200)),
    8: ("cnn_tn320.keras",     (320, 320)),
    9: ("cnn_tt460.keras",     (460, 460)),
}

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

@st.cache_resource(show_spinner=False)
def load_all_models(model_dir: str):
    """Load Keras models once and cache them."""
    models = {}
    missing = []
    for idx, (fname, _) in CLASS_CONFIGS.items():
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            missing.append(fpath)
            continue
        models[idx] = load_model(fpath, custom_objects={'mse': mse})
    return models, missing

MODELS, MISSING = load_all_models(model_dir)
if MISSING:
    with st.sidebar.expander("Missing models", expanded=False):
        for p in MISSING:
            st.markdown(f"- `{p}`")

def drawbox(img, label, a, b, c, d, color):
    # bbox [top,bottom,left,right]
    cv2.rectangle(img, (c, a), (d, b), color, 2)
    cv2.putText(img, label, (c, max(0, a - 10)), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 2)
    return img

def compute_iou_xyxy(box, boxes):
    x1, y1, x2, y2 = box
    xx1 = np.maximum(x1, boxes[:,0])
    yy1 = np.maximum(y1, boxes[:,1])
    xx2 = np.minimum(x2, boxes[:,2])
    yy2 = np.minimum(y2, boxes[:,3])
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    inter = w * h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    union = area1 + area2 - inter
    return np.where(union > 0, inter / union, 0.0)

def nms_fast(dets, iou_thr=0.5):
    if not dets:
        return []
    out = []
    by_class = {}
    for d in dets:
        by_class.setdefault(d['class_idx'], []).append(d)
    for cls, items in by_class.items():
        tb = np.array([d['bbox'] for d in items], dtype=np.int32)  # [t,b,l,r]
        boxes = np.stack([tb[:,2], tb[:,0], tb[:,3], tb[:,1]], axis=1).astype(np.float32)  # xyxy
        scores = np.array([d['score'] for d in items], dtype=np.float32)
        order = scores.argsort()[::-1]
        keep_idx = []
        while order.size > 0:
            i = order[0]
            keep_idx.append(i)
            if order.size == 1:
                break
            ious = compute_iou_xyxy(boxes[i], boxes[order[1:]])
            remain = np.where(ious < iou_thr)[0] + 1
            order = order[remain]
        out.extend([items[i] for i in keep_idx])
    return out

def merge_connected_boxes_by_class(detections, merge_iou_threshold):
    if not detections:
        return []
    merged = []
    by_class = {}
    for d in detections:
        by_class.setdefault(d['class_idx'], []).append(d)
    for cls, items in by_class.items():
        N = len(items)
        b = np.array([d['bbox'] for d in items], dtype=np.int32)  # [t,b,l,r]
        xyxy = np.stack([b[:,2], b[:,0], b[:,3], b[:,1]], axis=1).astype(np.float32)
        parent = np.arange(N)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, c):
            ra, rc = find(a), find(c)
            if ra != rc: parent[rc] = ra
        for i in range(N):
            if i+1 < N:
                ious = compute_iou_xyxy(xyxy[i], xyxy[i+1:])
                to_merge = np.where(ious > merge_iou_threshold)[0]
                for off in to_merge:
                    j = i + 1 + off
                    union(i, j)
        groups = {}
        for i in range(N):
            r = find(i)
            groups.setdefault(r, []).append(i)
        for gidxs in groups.values():
            tops   = int(np.min(b[gidxs, 0]))
            bots   = int(np.max(b[gidxs, 1]))
            lefts  = int(np.min(b[gidxs, 2]))
            rights = int(np.max(b[gidxs, 3]))
            max_score = float(np.max([items[k]['score'] for k in gidxs]))
            merged.append({"bbox": [tops, bots, lefts, rights], "class_idx": cls, "score": max_score})
    return merged

def object_det_from_image(img_bgr: np.ndarray, threshold=0.5, nms_threshold=0.4, merge_iou_threshold=0.3, step_size=None, input_size=(64,64), background_std_cut=4.0):
    assert img_bgr is not None and img_bgr.ndim == 3, "Input image must be BGR (H,W,3)"
    H, W = img_bgr.shape[:2]
    in_w, in_h = int(input_size[0]), int(input_size[1])
    detections = []

    t0 = time.perf_counter()
    tiny = cv2.resize(img_bgr, (min(128, W), min(128, H)), interpolation=cv2.INTER_AREA)

    prep_time = time.perf_counter()
    for class_idx, (fname, (box_h, box_w)) in CLASS_CONFIGS.items():
        model = MODELS.get(class_idx, None)
        if model is None:
            continue
        stride = step_size if step_size is not None else max(16, min(box_h, box_w) // 4)
        ys = np.arange(0, max(1, H - box_h + 1), stride, dtype=np.int32)
        xs = np.arange(0, max(1, W - box_w + 1), stride, dtype=np.int32)
        if ys.size == 0 or xs.size == 0:
            continue
        coords = np.stack(np.meshgrid(ys, xs, indexing='ij'), axis=-1).reshape(-1, 2)

        scale_y = tiny.shape[0] / H
        scale_x = tiny.shape[1] / W
        ty = (coords[:,0] * scale_y).astype(np.int32)
        tx = (coords[:,1] * scale_x).astype(np.int32)
        th = max(1, int(box_h * scale_y))
        tw = max(1, int(box_w * scale_x))
        ty2 = np.minimum(ty + th, tiny.shape[0])
        tx2 = np.minimum(tx + tw, tiny.shape[1])

        keep_mask = np.zeros(len(coords), dtype=bool)
        CHUNK = 2048
        for start in range(0, len(coords), CHUNK):
            end = min(len(coords), start + CHUNK)
            stds = np.empty(end - start, dtype=np.float32)
            for i in range(start, end):
                roi = tiny[ty[i]:ty2[i], tx[i]:tx2[i]]
                if roi.size == 0:
                    stds[i - start] = 0.0
                    continue
                roi_small = cv2.resize(roi, (16,16), interpolation=cv2.INTER_AREA)
                stds[i - start] = float(roi_small.std())
            keep_mask[start:end] = stds >= background_std_cut

        coords_kept = coords[keep_mask]
        if coords_kept.size == 0:
            continue

        patches = []
        anchors = []
        for (y, x) in coords_kept:
            patch = img_bgr[y:y+box_h, x:x+box_w]
            if patch.shape[0] != box_h or patch.shape[1] != box_w:
                continue
            patch_in = cv2.resize(patch, (in_w, in_h), interpolation=cv2.INTER_AREA)
            patches.append(patch_in)
            anchors.append((y, x))

        if not patches:
            continue

        X = np.asarray(patches, dtype=np.uint8)
        y_out = model.predict(X, batch_size=128, verbose=0)
        y_out = np.asarray(y_out)
        if y_out.ndim == 2 and y_out.shape[1] >= 2:
            scores = y_out[:,1]
        else:
            scores = y_out.reshape(-1)

        pass_idx = np.where(scores > threshold)[0]
        for k in pass_idx:
            y, x = anchors[k]
            detections.append({"bbox": [int(y), int(y + box_h), int(x), int(x + box_w)], "score": float(scores[k]), "class_idx": class_idx})
    infer_time = time.perf_counter()

    nms_dets   = nms_fast(detections, iou_thr=nms_threshold)
    merged     = merge_connected_boxes_by_class(nms_dets, merge_iou_threshold)
    post_time  = time.perf_counter()

    out = img_bgr.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255), (255,0,255),
              (255,255,0), (128,128,0), (0,128,128), (128,0,128), (100,100,100), (200,100,50)]
    for det in merged:
        a,b,c,d = det["bbox"]
        cls = det["class_idx"]
        label = f"{CLASS_LABEL[cls]}: {det['score']:.2f}"
        color = colors[cls % len(colors)]
        drawbox(out, label, a, b, c, d, color)

    timings = {
        "prep_ms":   (prep_time - t0) * 1000,
        "infer_ms":  (infer_time - prep_time) * 1000,
        "post_ms":   (post_time - infer_time) * 1000,
        "total_ms":  (post_time - t0) * 1000,
        "num_windows": len(detections),
        "num_after_nms": len(nms_dets),
        "num_merged": len(merged)
    }
    return out, merged, timings

st.title("🧪 AI Detector")

tab1, tab2 = st.tabs(["Run Detector", "Advanced Parameters"])

with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","tif","tiff","bmp"])
    path_input = st.text_input("...or enter an image path", value="", help="Useful on Colab/Drive")
    colA, colB, colC = st.columns(3)
    with colA:
        threshold = st.slider("Score threshold", 0.0, 1.0, 0.50, 0.01)
        nms_thr   = st.slider("NMS IoU", 0.0, 1.0, 0.40, 0.01)
    with colB:
        merge_thr = st.slider("Merge IoU", 0.0, 1.0, 0.30, 0.01)
        bg_std    = st.slider("Background std cutoff", 0.0, 20.0, 4.0, 0.5, help="Higher skips more flat patches (faster, but may miss faint objects)")
    with colC:
        step_size = st.number_input("Override stride (px)", min_value=0, value=0, help="0 = automatic per class")
        in_w = st.number_input("Model input width", 32, 512, 64, step=16)
        in_h = st.number_input("Model input height", 32, 512, 64, step=16)

    run = st.button("Run detection", type="primary", use_container_width=True)

    img_bgr = None
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif path_input:
        img_bgr = cv2.imread(path_input)

    if run:
        if img_bgr is None:
            st.warning("Please upload an image or enter a valid path.")
        elif not MODELS:
            st.error("No models loaded. Check your model directory.")
        else:
            with st.spinner("Running detection..."):
                out, dets, t = object_det_from_image(
                    img_bgr,
                    threshold=threshold,
                    nms_threshold=nms_thr,
                    merge_iou_threshold=merge_thr,
                    step_size=(None if step_size <= 0 else int(step_size)),
                    input_size=(int(in_w), int(in_h)),
                    background_std_cut=float(bg_std)
                )

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("Detections")
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("### Stats")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total (ms)", f"{t['total_ms']:.1f}")
            c2.metric("Prep / Infer / Post (ms)", f"{t['prep_ms']:.1f} / {t['infer_ms']:.1f} / {t['post_ms']:.1f}")
            c3.metric("Windows", f"{t['num_windows']}")
            c4.metric("NMS→Merged", f"{t['num_after_nms']} → {t['num_merged']}")

            # Download
            ok, buf = cv2.imencode(".png", out)
            if ok:
                st.download_button("Download result PNG", data=buf.tobytes(), file_name="ai_detector_result.png",
                                   mime="image/png", use_container_width=True)

with tab2:
    st.markdown("""
**Tips for more speed**
- Put a `Rescaling(1/255.)` layer at the start of each `.keras` model so we can feed `uint8` directly (no Python-side normalization).
- Use GPU with mixed precision if available.
- Increase stride (Override stride) for large kernels (e.g., 650×650) to cut window count.
- Raise *Background std cutoff* to skip flat tiles.
- If your dataset is very large, consider a small proposal model to gate which class model to run per location.
""")
    st.code("""
# Example: add this layer into your Keras model so preprocessing is on-graph
from keras.layers import Rescaling
...
x = Rescaling(1./255.0)(inputs)
""", language="python")
