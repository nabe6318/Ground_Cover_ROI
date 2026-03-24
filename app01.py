# -*- coding: utf-8 -*-
import math
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates

# ------------------------------------------------------------
# ページ設定
# ------------------------------------------------------------
st.set_page_config(page_title="Ground Cover ROI + Scale + Grid", layout="wide")

st.title("🌿 RGB画像からROI切り出し・縮尺設定・グリッド被覆率計算")
st.markdown(
    """
RGBドローン画像から、緑色植物の被覆率を計算します。

**流れ**
1. 元画像で **4点クリック** して ROI を切り出す  
2. 切り出した画像で **2点クリック** して縮尺を設定  
3. 任意サイズのグリッドで被覆率を計算し、ヒートマップ表示
"""
)

# ------------------------------------------------------------
# 基本関数
# ------------------------------------------------------------
def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def resize_if_needed(rgb, max_size=1800):
    h, w = rgb.shape[:2]
    max_edge = max(h, w)
    if max_edge <= max_size:
        return rgb, 1.0
    scale = max_size / max_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    pil_img = Image.fromarray(rgb)
    pil_img = pil_img.resize((new_w, new_h))
    return np.array(pil_img), scale

def rgb_to_hsv_np(rgb):
    img_pil = Image.fromarray(rgb).convert("HSV")
    hsv = np.array(img_pil).astype(np.float32)
    # H(0-255), S(0-255), V(0-255)
    return hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

def calc_exg(rgb):
    """
    論文準拠の正規化ExG: (2G - R - B) / (R + G + B) 
    """
    rgbf = rgb.astype(np.float32) / 255.0
    r = rgbf[:, :, 0]
    g = rgbf[:, :, 1]
    b = rgbf[:, :, 2]
    
    numerator = 2 * g - r - b
    denominator = r + g + b
    
    # R=G=B=0（分母0）の場合は ExG=0 と処理 
    exg = np.where(denominator != 0, numerator / denominator, 0.0)
    return exg

def calc_vari(rgb):
    rgbf = rgb.astype(np.float32) / 255.0
    r = rgbf[:, :, 0]
    g = rgbf[:, :, 1]
    b = rgbf[:, :, 2]
    denom = g + r - b
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
    vari = (g - r) / denom
    return np.nan_to_num(vari, nan=-9999)

def make_mask_hsv(rgb, h_min, h_max, s_min, v_min):
    h, s, v = rgb_to_hsv_np(rgb)
    # 0-255スケールのH, S, Vに対応
    if h_min <= h_max:
        h_mask = (h >= h_min) & (h <= h_max)
    else:
        h_mask = (h >= h_min) | (h <= h_max)
    mask = h_mask & (s >= s_min) & (v >= v_min)
    return mask

def make_mask_exg(rgb, exg_threshold):
    exg = calc_exg(rgb)
    return exg >= exg_threshold

def make_mask_vari(rgb, vari_threshold):
    vari = calc_vari(rgb)
    return vari >= vari_threshold

def calc_cover_rate(mask):
    if mask.size == 0:
        return 0.0
    return float(mask.sum()) / float(mask.size) * 100.0

def create_binary_image(mask):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask] = [0, 255, 0]
    return out

def mask_to_overlay(rgb, mask, alpha=0.40):
    overlay = rgb.copy().astype(np.float32)
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    overlay[mask] = overlay[mask] * (1 - alpha) + green[mask] * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)

def pixel_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def draw_points_and_lines(rgb, points, close_polygon=False, point_color="red", line_color="red"):
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    r = max(4, int(min(img.size) * 0.007))

    for p in points:
        x, y = p
        draw.ellipse((x-r, y-r, x+r, y+r), outline=point_color, width=3)

    if len(points) >= 2:
        for i in range(len(points) - 1):
            draw.line((points[i][0], points[i][1], points[i+1][0], points[i+1][1]), fill=line_color, width=3)

    if close_polygon and len(points) >= 4:
        draw.line((points[-1][0], points[-1][1], points[0][0], points[0][1]), fill=line_color, width=3)

    return np.array(img)

def crop_roi_by_4points(rgb, points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = max(0, int(min(xs)))
    x_max = min(rgb.shape[1], int(max(xs)))
    y_min = max(0, int(min(ys)))
    y_max = min(rgb.shape[0], int(max(ys)))

    cropped = rgb[y_min:y_max, x_min:x_max].copy()
    return cropped, x_min, y_min, x_max, y_max

def compute_grid_cover(mask, meters_per_pixel, grid_size_m=1.0, drop_partial=False):
    h, w = mask.shape
    grid_px = grid_size_m / meters_per_pixel
    grid_px_int = max(1, int(round(grid_px)))

    if drop_partial:
        n_rows = h // grid_px_int
        n_cols = w // grid_px_int
    else:
        n_rows = math.ceil(h / grid_px_int)
        n_cols = math.ceil(w / grid_px_int)

    grid_values = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    records = []

    for r in range(n_rows):
        y0 = r * grid_px_int
        y1 = min((r + 1) * grid_px_int, h)

        for c in range(n_cols):
            x0 = c * grid_px_int
            x1 = min((c + 1) * grid_px_int, w)

            if y0 >= h or x0 >= w:
                continue

            cell = mask[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            cover = float(cell.sum()) / float(cell.size) * 100.0
            grid_values[r, c] = cover

            records.append({
                "row": r, "col": c, "cover_percent": cover,
                "x0_m": x0 * meters_per_pixel, "y0_m": y0 * meters_per_pixel
            })

    return grid_values, pd.DataFrame(records), grid_px_int

def create_heatmap_figure(grid_values, title="Grid Cover Heatmap (%)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid_values, aspect="equal", cmap="YlGn", vmin=0, vmax=100)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Cover (%)")
    return fig

def draw_grid_on_image(rgb, grid_px):
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]
    for x in range(0, w, grid_px): draw.line((x, 0, x, h), fill=(255, 255, 0), width=1)
    for y in range(0, h, grid_px): draw.line((0, y, w, y), fill=(255, 255, 0), width=1)
    return np.array(img)

def pil_to_png_bytes(np_img):
    bio = io.BytesIO()
    Image.fromarray(np_img).save(bio, format="PNG")
    return bio.getvalue()

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
defaults = {
    "roi_points": [], "roi_last_click": None, "roi_canvas_key": 0,
    "scale_points": [], "scale_last_click": None, "scale_canvas_key": 0,
    "last_uploaded_name": None, "cropped_roi_image": None, "cropped_roi_bounds": None,
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("設定")
uploaded_file = st.sidebar.file_uploader("RGB画像をアップロード", type=["jpg", "jpeg", "png", "tif", "tiff"])

# デフォルトをExGに設定
method = st.sidebar.selectbox("緑抽出方法", ["ExG", "HSV(HSB)", "VARI"], index=0)

resize_option = st.sidebar.checkbox("大きな画像を縮小して表示・解析", value=True)
max_size = st.sidebar.number_input("表示用最大辺ピクセル数", 600, 5000, 1800, 100)

st.sidebar.markdown("---")
st.sidebar.subheader("縮尺設定")
known_distance_m = st.sidebar.number_input("既知の長さ (m)", min_value=0.001, value=1.0, step=0.1, format="%.3f")

st.sidebar.subheader("グリッド設定")
grid_size_m = st.sidebar.number_input("グリッドサイズ (m)", min_value=0.05, value=1.0, step=0.1, format="%.2f")
drop_partial = st.sidebar.checkbox("端の不完全グリッドを除外", value=False)

# ------------------------------------------------------------
# Main Process
# ------------------------------------------------------------
if uploaded_file is not None:
    current_name = uploaded_file.name
    if st.session_state.last_uploaded_name != current_name:
        for k in ["roi_points", "scale_points"]: st.session_state[k] = []
        st.session_state.cropped_roi_image = None
        st.session_state.last_uploaded_name = current_name

    rgb_orig = load_image(uploaded_file)
    rgb_disp, _ = resize_if_needed(rgb_orig, max_size=max_size) if resize_option else (rgb_orig.copy(), 1.0)

    st.subheader("1. 元画像")
    st.write(f"表示サイズ: {rgb_disp.shape[1]} × {rgb_disp.shape[0]} px")

    # STEP 2: ROI Selection
    st.markdown("---")
    st.subheader("2. ROIを4点クリックして切り出し")
    roi_prev = draw_points_and_lines(rgb_disp, st.session_state.roi_points, close_polygon=True)
    roi_click = streamlit_image_coordinates(Image.fromarray(roi_prev), key=f"roi_{st.session_state.roi_canvas_key}")

    if roi_click:
        p = (int(roi_click["x"]), int(roi_click["y"]))
        if p != st.session_state.roi_last_click:
            if len(st.session_state.roi_points) >= 4: st.session_state.roi_points = [p]
            else: st.session_state.roi_points.append(p)
            st.session_state.roi_last_click = p

    rc1, rc2, rc3 = st.columns(3)
    if rc1.button("ROI リセット"):
        st.session_state.roi_points = []
        st.session_state.roi_canvas_key += 1
        st.rerun()
    if rc3.button("ROI 切り出し確定"):
        if len(st.session_state.roi_points) == 4:
            st.session_state.cropped_roi_image, *_ = crop_roi_by_4points(rgb_disp, st.session_state.roi_points)
            st.session_state.scale_points = []
            st.rerun()

    # STEP 3-6: ROI Analysis
    if st.session_state.cropped_roi_image is not None:
        cropped_rgb = st.session_state.cropped_roi_image
        st.markdown("---")
        st.subheader("3. 切り出しROI画像と抽出設定")
        
        if method == "ExG":
            # 論文値を参考に、スライダー初期値を設定 
            exg_t = st.sidebar.slider("ExG threshold", -1.0, 1.0, 0.10, 0.01)
            mask = make_mask_exg(cropped_rgb, exg_t)
        elif method == "HSV(HSB)":
            h_min = st.sidebar.slider("Hue 最小", 0, 255, 30)
            h_max = st.sidebar.slider("Hue 最大", 0, 255, 90)
            s_min = st.sidebar.slider("Saturation 最小", 0, 255, 50)
            v_min = st.sidebar.slider("Value 最小", 0, 255, 40)
            mask = make_mask_hsv(cropped_rgb, h_min, h_max, s_min, v_min)
        else:
            vari_t = st.sidebar.slider("VARI threshold", -1.0, 1.0, 0.00, 0.01)
            mask = make_mask_vari(cropped_rgb, vari_t)

        overlay = mask_to_overlay(cropped_rgb, mask)
        st.metric("ROI全体被覆率 (%)", f"{calc_cover_rate(mask):.2f}")
        st.image(overlay, caption="抽出プレビュー (緑色判定部)", use_container_width=True)

        # STEP 5: Scale
        st.markdown("---")
        st.subheader("5. ROI画像上で2点クリックして縮尺設定")
        scale_prev = draw_points_and_lines(cropped_rgb, st.session_state.scale_points)
        scale_click = streamlit_image_coordinates(Image.fromarray(scale_prev), key=f"scale_{st.session_state.scale_canvas_key}")

        if scale_click:
            p = (int(scale_click["x"]), int(scale_click["y"]))
            if p != st.session_state.scale_last_click:
                if len(st.session_state.scale_points) >= 2: st.session_state.scale_points = [p]
                else: st.session_state.scale_points.append(p)
                st.session_state.scale_last_click = p

        if st.button("縮尺リセット"):
            st.session_state.scale_points = []
            st.session_state.scale_canvas_key += 1
            st.rerun()

        if len(st.session_state.scale_points) == 2:
            pix_len = pixel_distance(st.session_state.scale_points[0], st.session_state.scale_points[1])
            m_per_px = known_distance_m / pix_len
            grid_vals, grid_df, g_px = compute_grid_cover(mask, m_per_px, grid_size_m, drop_partial)

            st.markdown("---")
            st.subheader("6. グリッド被覆率結果")
            g1, g2 = st.columns(2)
            g1.image(draw_grid_on_image(overlay, g_px), caption="グリッド表示")
            g2.pyplot(create_heatmap_figure(grid_vals, f"{grid_size_m}m Grid Heatmap"))
            
            st.download_button("データCSVを保存", data=grid_df.to_csv(index=False).encode("utf-8-sig"), file_name="result.csv")
else:
    st.info("サイドバーから画像をアップロードしてください。")
