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
st.set_page_config(page_title="Ground Cover Analysis Tool", layout="wide")

st.markdown("### 🌿 RGB画像 植被率解析アプリ")
st.markdown("*RGB画像からROIの切り出し、各種指標による植生抽出、グリッドごとの被覆率計算を行います。*")

# ------------------------------------------------------------
# 解析ロジック（各種指標）
# ------------------------------------------------------------

def calc_exg(rgb):
    """Excess Green Index (ExG): (2G - R - B) / (R + G + B) """
    rgbf = rgb.astype(np.float32) / 255.0
    r, g, b = rgbf[:,:,0], rgbf[:,:,1], rgbf[:,:,2]
    numerator = 2 * g - r - b
    denominator = r + g + b
    # 分母0（黒色画素）の場合は0を返す [cite: 65]
    return np.where(denominator != 0, numerator / denominator, 0.0)

def calc_vari(rgb):
    """Visible Resistant Vegetative Index (VARI): (G - R) / (G + R - B)"""
    rgbf = rgb.astype(np.float32) / 255.0
    r, g, b = rgbf[:,:,0], rgbf[:,:,1], rgbf[:,:,2]
    numerator = g - r
    denominator = g + r - b
    return np.where(denominator != 0, numerator / denominator, 0.0)

def rgb_to_hsv_np(rgb):
    """HSV変換 (色相・彩度・明度)"""
    img_pil = Image.fromarray(rgb).convert("HSV")
    hsv = np.array(img_pil).astype(np.float32)
    return hsv[:,:,0], hsv[:,:,1], hsv[:,:,2] # H(0-255), S(0-255), V(0-255)

# ------------------------------------------------------------
# マスク作成関数
# ------------------------------------------------------------

def make_mask(rgb, method, params):
    if method == "ExG":
        val = calc_exg(rgb)
        return val >= params["threshold"]
    
    elif method == "VARI":
        val = calc_vari(rgb)
        return val >= params["threshold"]
    
    elif method == "HSV":
        h, s, v = rgb_to_hsv_np(rgb)
        # Hの範囲指定（0-255スケール）
        h_mask = (h >= params["h_min"]) & (h <= params["h_max"])
        s_mask = s >= params["s_min"]
        v_mask = v >= params["v_min"]
        return h_mask & s_mask & v_mask
    return np.zeros(rgb.shape[:2], dtype=bool)

# ------------------------------------------------------------
# 基本関数・ユーティリティ
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
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = Image.fromarray(rgb).resize((new_w, new_h))
    return np.array(pil_img), scale

def calc_cover_rate(mask):
    if mask.size == 0: return 0.0
    return float(mask.sum()) / float(mask.size) * 100.0

def mask_to_overlay(rgb, mask, alpha=0.40):
    overlay = rgb.copy().astype(np.float32)
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255 # 判定箇所を緑色でハイライト
    overlay[mask] = overlay[mask] * (1 - alpha) + green[mask] * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)

def pixel_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def draw_points_and_lines(rgb, points, close_polygon=False):
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    r = max(4, int(min(img.size) * 0.007))
    for p in points:
        draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), outline="red", width=3)
    if len(points) >= 2:
        for i in range(len(points) - 1):
            draw.line((points[i][0], points[i][1], points[i+1][0], points[i+1][1]), fill="red", width=3)
    if close_polygon and len(points) >= 4:
        draw.line((points[-1][0], points[-1][1], points[0][0], points[0][1]), fill="red", width=3)
    return np.array(img)

def crop_roi_by_4points(rgb, points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max = max(0, int(min(xs))), min(rgb.shape[1], int(max(xs)))
    y_min, y_max = max(0, int(min(ys))), min(rgb.shape[0], int(max(ys)))
    return rgb[y_min:y_max, x_min:x_max].copy()

def compute_grid_cover(mask, meters_per_pixel, grid_size_m=1.0):
    h, w = mask.shape
    grid_px = max(1, int(round(grid_size_m / meters_per_pixel)))
    n_rows, n_cols = math.ceil(h / grid_px), math.ceil(w / grid_px)
    grid_values = np.full((n_rows, n_cols), np.nan)
    records = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * grid_px, min((r + 1) * grid_px, h)
            x0, x1 = c * grid_px, min((c + 1) * grid_px, w)
            cell = mask[y0:y1, x0:x1]
            if cell.size > 0:
                cover = (cell.sum() / cell.size) * 100.0
                grid_values[r, c] = cover
                records.append({"row": r, "col": c, "cover_percent": cover})
    return grid_values, pd.DataFrame(records), grid_px

# ------------------------------------------------------------
# Session State & Sidebar UI
# ------------------------------------------------------------
if "roi_points" not in st.session_state:
    st.session_state.update({"roi_points":[], "roi_last_click":None, "roi_canvas_key":0,
                             "scale_points":[], "scale_last_click":None, "scale_canvas_key":0,
                             "cropped_roi_image":None})

st.sidebar.header("1. 解析設定")
uploaded_file = st.sidebar.file_uploader("RGB画像をアップロード", type=["jpg", "png", "tif"])

method = st.sidebar.selectbox("抽出指標を選択", ["ExG", "HSV", "VARI"])
params = {}

if method in ["ExG", "VARI"]:
    params["threshold"] = st.sidebar.slider(f"{method} 閾値", -1.00, 1.00, 0.10, 0.01)
else:
    params["h_min"] = st.sidebar.slider("Hue 最小 (0-255)", 0, 255, 40)
    params["h_max"] = st.sidebar.slider("Hue 最大 (0-255)", 0, 255, 90)
    params["s_min"] = st.sidebar.slider("Saturation 最小 (0-255)", 0, 255, 30)
    params["v_min"] = st.sidebar.slider("Value 最小 (0-255)", 0, 255, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("2. 縮尺・グリッド設定")
known_dist = st.sidebar.number_input("既知の長さ (m)", 0.001, 100.0, 1.0, step=0.1)
grid_size = st.sidebar.number_input("グリッドサイズ (m)", 0.01, 10.0, 1.0, step=0.1)

# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
if uploaded_file:
    rgb_orig = load_image(uploaded_file)
    rgb_disp, _ = resize_if_needed(rgb_orig)

    st.subheader("STEP 1: ROI（解析範囲）の指定")
    roi_view = draw_points_and_lines(rgb_disp, st.session_state.roi_points, close_polygon=True)
    roi_click = streamlit_image_coordinates(Image.fromarray(roi_view), key=f"roi_{st.session_state.roi_canvas_key}")

    if roi_click:
        curr = (int(roi_click["x"]), int(roi_click["y"]))
        if curr != st.session_state.roi_last_click:
            if len(st.session_state.roi_points) >= 4: st.session_state.roi_points = [curr]
            else: st.session_state.roi_points.append(curr)
            st.session_state.roi_last_click = curr
            st.rerun()

    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("ROIリセット"):
        st.session_state.roi_points = []
        st.session_state.cropped_roi_image = None
        st.session_state.roi_canvas_key += 1
        st.rerun()

    if len(st.session_state.roi_points) == 4 and col_btn2.button("この範囲で確定"):
        st.session_state.cropped_roi_image = crop_roi_by_4points(rgb_disp, st.session_state.roi_points)
        st.rerun()

    if st.session_state.cropped_roi_image is not None:
        cropped_rgb = st.session_state.cropped_roi_image
        mask = make_mask(cropped_rgb, method, params)
        overlay = mask_to_overlay(cropped_rgb, mask)

        st.markdown("---")
        st.subheader(f"STEP 2: 抽出結果 ({method}) と縮尺設定")
        
        col_img, col_met = st.columns([2, 1])
        with col_img:
            st.image(overlay, caption="抽出プレビュー（緑色が判定箇所）", use_container_width=True)
        with col_met:
            st.metric("ROI内 被覆率", f"{calc_cover_rate(mask):.2f} %")
            st.info("画像内の「1メートル」など既知の長さを2点クリックしてください。")

        scale_view = draw_points_and_lines(cropped_rgb, st.session_state.scale_points)
        scale_click = streamlit_image_coordinates(Image.fromarray(scale_view), key=f"scale_{st.session_state.scale_canvas_key}")

        if scale_click:
            sc_curr = (int(scale_click["x"]), int(scale_click["y"]))
            if sc_curr != st.session_state.scale_last_click:
                if len(st.session_state.scale_points) >= 2: st.session_state.scale_points = [sc_curr]
                else: st.session_state.scale_points.append(sc_curr)
                st.session_state.scale_last_click = sc_curr
                st.rerun()

        if len(st.session_state.scale_points) == 2:
            pix_len = pixel_distance(st.session_state.scale_points[0], st.session_state.scale_points[1])
            m_per_px = known_dist / pix_len
            grid_vals, grid_df, g_px = compute_grid_cover(mask, m_per_px, grid_size)

            st.markdown("---")
            st.subheader(f"STEP 3: グリッド解析結果 ({grid_size}m四方)")
            res_col1, res_col2 = st.columns(2)
            
            # グリッド描画
            img_grid = Image.fromarray(overlay.copy())
            draw_g = ImageDraw.Draw(img_grid)
            for x in range(0, overlay.shape[1], g_px): draw_g.line((x, 0, x, overlay.shape[0]), fill="yellow")
            for y in range(0, overlay.shape[0], g_px): draw_g.line((0, y, overlay.shape[1], y), fill="yellow")
            res_col1.image(img_grid, caption="解析グリッド")

            # ヒートマップ
            fig, ax = plt.subplots()
            im = ax.imshow(grid_vals, cmap="YlGn", vmin=0, vmax=100)
            plt.colorbar(im, label="Cover %")
            res_col2.pyplot(fig)

            st.download_button("結果CSVを保存", grid_df.to_csv(index=False).encode("utf-8-sig"), "result.csv")
else:
    st.info("画像をアップロードしてください。")
