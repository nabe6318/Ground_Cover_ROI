# -*- coding: utf-8 -*-
import math
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates

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
    rgb = rgb.astype(np.float32) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    cmax = np.max(rgb, axis=2)
    cmin = np.min(rgb, axis=2)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta != 0

    idx = mask & (cmax == r)
    h[idx] = (60 * ((g[idx] - b[idx]) / delta[idx]) + 360) % 360

    idx = mask & (cmax == g)
    h[idx] = (60 * ((b[idx] - r[idx]) / delta[idx]) + 120) % 360

    idx = mask & (cmax == b)
    h[idx] = (60 * ((r[idx] - g[idx]) / delta[idx]) + 240) % 360

    s = np.zeros_like(cmax)
    s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    v = cmax
    return h, s, v

def calc_exg(rgb):
    rgbf = rgb.astype(np.float32) / 255.0
    r = rgbf[:, :, 0]
    g = rgbf[:, :, 1]
    b = rgbf[:, :, 2]
    return 2 * g - r - b

def calc_vari(rgb):
    rgbf = rgb.astype(np.float32)
    r = rgbf[:, :, 0]
    g = rgbf[:, :, 1]
    b = rgbf[:, :, 2]
    denom = g + r - b
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
    vari = (g - r) / denom
    return np.nan_to_num(vari, nan=-9999)

def make_mask_hsv(rgb, h_min, h_max, s_min, v_min):
    h, s, v = rgb_to_hsv_np(rgb)
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
        return np.nan
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
    """
    4点を含む外接矩形で切り出し
    points: [(x, y), ...] in current image coordinates
    """
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

    if grid_px <= 1:
        grid_px = 1.0

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
                "row": r,
                "col": c,
                "x0_px": x0,
                "y0_px": y0,
                "x1_px": x1,
                "y1_px": y1,
                "cell_width_px": x1 - x0,
                "cell_height_px": y1 - y0,
                "grid_size_m": grid_size_m,
                "cover_percent": cover,
                "x0_m": x0 * meters_per_pixel,
                "y0_m": y0 * meters_per_pixel,
                "x1_m": x1 * meters_per_pixel,
                "y1_m": y1 * meters_per_pixel,
            })

    return grid_values, pd.DataFrame(records), grid_px_int

def create_heatmap_figure(grid_values, title="Grid Cover Heatmap (%)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid_values, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cover (%)")
    return fig

def draw_grid_on_image(rgb, grid_px):
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]

    for x in range(0, w, grid_px):
        draw.line((x, 0, x, h), fill=(255, 255, 0), width=1)
    for y in range(0, h, grid_px):
        draw.line((0, y, w, y), fill=(255, 255, 0), width=1)

    return np.array(img)

def pil_to_png_bytes(np_img):
    pil_img = Image.fromarray(np_img)
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    return bio.getvalue()

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
defaults = {
    "roi_points": [],
    "roi_last_click": None,
    "roi_canvas_key": 0,
    "scale_points": [],
    "scale_last_click": None,
    "scale_canvas_key": 0,
    "last_uploaded_name": None,
    "cropped_roi_image": None,
    "cropped_roi_bounds": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("設定")

uploaded_file = st.sidebar.file_uploader(
    "RGB画像をアップロード",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

method = st.sidebar.selectbox("緑抽出方法", ["HSV(HSB)", "ExG", "VARI"])

resize_option = st.sidebar.checkbox("大きな画像を縮小して表示・解析", value=True)
max_size = st.sidebar.number_input("表示用最大辺ピクセル数", 600, 5000, 1800, 100)

st.sidebar.markdown("---")
st.sidebar.subheader("縮尺設定")
known_distance_m = st.sidebar.number_input(
    "Known distance (m)",
    min_value=0.001,
    value=1.0,
    step=0.1,
    format="%.3f"
)

st.sidebar.subheader("グリッド設定")
grid_size_m = st.sidebar.number_input(
    "Grid size (m)",
    min_value=0.05,
    value=1.0,
    step=0.1,
    format="%.2f"
)
drop_partial = st.sidebar.checkbox("端の不完全グリッドを除外", value=False)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if uploaded_file is not None:
    current_uploaded_name = uploaded_file.name
    if st.session_state.last_uploaded_name != current_uploaded_name:
        st.session_state.roi_points = []
        st.session_state.roi_last_click = None
        st.session_state.roi_canvas_key += 1
        st.session_state.scale_points = []
        st.session_state.scale_last_click = None
        st.session_state.scale_canvas_key += 1
        st.session_state.cropped_roi_image = None
        st.session_state.cropped_roi_bounds = None
        st.session_state.last_uploaded_name = current_uploaded_name

    rgb_original = load_image(uploaded_file)
    original_h, original_w = rgb_original.shape[:2]

    if resize_option:
        rgb_display, display_scale = resize_if_needed(rgb_original, max_size=max_size)
    else:
        rgb_display = rgb_original.copy()
        display_scale = 1.0

    disp_h, disp_w = rgb_display.shape[:2]

    st.subheader("1. 元画像")
    st.write(f"元画像サイズ: {original_w} × {original_h} px")
    st.write(f"表示・解析サイズ: {disp_w} × {disp_h} px")

    # ========================================================
    # ROI selection
    # ========================================================
    st.markdown("---")
    st.subheader("2. ROIを4点クリックして切り出し")

    roi_preview = draw_points_and_lines(
        rgb_display,
        st.session_state.roi_points,
        close_polygon=True,
        point_color="red",
        line_color="red"
    )

    roi_clicked = streamlit_image_coordinates(
        Image.fromarray(roi_preview),
        key=f"roi_click_image_{st.session_state.roi_canvas_key}"
    )

    if roi_clicked is not None:
        current_click = (int(roi_clicked["x"]), int(roi_clicked["y"]))
        if current_click != st.session_state.roi_last_click:
            if len(st.session_state.roi_points) >= 4:
                st.session_state.roi_points = [current_click]
            else:
                st.session_state.roi_points.append(current_click)
            st.session_state.roi_last_click = current_click

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        if st.button("ROI 4点を完全リセット"):
            st.session_state.roi_points = []
            st.session_state.roi_last_click = None
            st.session_state.roi_canvas_key += 1
            st.session_state.cropped_roi_image = None
            st.session_state.cropped_roi_bounds = None
            st.session_state.scale_points = []
            st.session_state.scale_last_click = None
            st.session_state.scale_canvas_key += 1
            st.rerun()
    with rc2:
        if st.button("ROI 最後の1点を削除"):
            if len(st.session_state.roi_points) > 0:
                st.session_state.roi_points = st.session_state.roi_points[:-1]
            st.session_state.roi_last_click = None
            st.session_state.roi_canvas_key += 1
            st.session_state.cropped_roi_image = None
            st.session_state.cropped_roi_bounds = None
            st.session_state.scale_points = []
            st.session_state.scale_last_click = None
            st.session_state.scale_canvas_key += 1
            st.rerun()
    with rc3:
        if st.button("ROI を切り出して保存用に確定"):
            if len(st.session_state.roi_points) == 4:
                cropped, x_min, y_min, x_max, y_max = crop_roi_by_4points(
                    rgb_display, st.session_state.roi_points
                )
                st.session_state.cropped_roi_image = cropped
                st.session_state.cropped_roi_bounds = (x_min, y_min, x_max, y_max)
                st.session_state.scale_points = []
                st.session_state.scale_last_click = None
                st.session_state.scale_canvas_key += 1
                st.rerun()

    if len(st.session_state.roi_points) > 0:
        st.write("ROI points:")
        for i, p in enumerate(st.session_state.roi_points, 1):
            st.write(f"{i}点目: x={p[0]}, y={p[1]}")

    # ========================================================
    # ROI cropped image
    # ========================================================
    if st.session_state.cropped_roi_image is not None:
        cropped_rgb = st.session_state.cropped_roi_image
        roi_h, roi_w = cropped_rgb.shape[:2]

        st.markdown("---")
        st.subheader("3. 切り出しROI画像")
        st.write(f"ROI画像サイズ: {roi_w} × {roi_h} px")

        b1, b2 = st.columns(2)
        with b1:
            st.image(cropped_rgb, caption="切り出しROI画像", use_container_width=True)
        with b2:
            st.download_button(
                "ROI画像をPNG保存",
                data=pil_to_png_bytes(cropped_rgb),
                file_name="cropped_roi.png",
                mime="image/png"
            )

        # ----------------------------------------------------
        # Vegetation mask on ROI
        # ----------------------------------------------------
        if method == "HSV(HSB)":
            st.sidebar.subheader("HSVしきい値")
            h_min = st.sidebar.slider("Hue 最小", 0, 360, 35)
            h_max = st.sidebar.slider("Hue 最大", 0, 360, 140)
            s_min = st.sidebar.slider("Saturation 最小", 0.0, 1.0, 0.20, 0.01)
            v_min = st.sidebar.slider("Value 最小", 0.0, 1.0, 0.15, 0.01)
            mask = make_mask_hsv(cropped_rgb, h_min, h_max, s_min, v_min)

        elif method == "ExG":
            st.sidebar.subheader("ExGしきい値")
            exg_threshold = st.sidebar.slider("ExG threshold", -1.0, 2.0, 0.05, 0.01)
            mask = make_mask_exg(cropped_rgb, exg_threshold)

        else:
            st.sidebar.subheader("VARIしきい値")
            vari_threshold = st.sidebar.slider("VARI threshold", -1.0, 1.0, 0.00, 0.01)
            mask = make_mask_vari(cropped_rgb, vari_threshold)

        overlay = mask_to_overlay(cropped_rgb, mask)
        binary = create_binary_image(mask)
        total_cover = calc_cover_rate(mask)

        st.subheader("4. ROI全体の被覆率")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("ROI全体被覆率 (%)", f"{total_cover:.2f}")
        mc2.metric("緑画素数", f"{int(mask.sum()):,}")
        mc3.metric("総画素数", f"{int(mask.size):,}")

        cc1, cc2, cc3 = st.columns(3)
        cc1.image(cropped_rgb, caption="ROI元画像", use_container_width=True)
        cc2.image(binary, caption="ROI抽出マスク", use_container_width=True)
        cc3.image(overlay, caption="ROI重ね合わせ", use_container_width=True)

        # ----------------------------------------------------
        # Scale on ROI
        # ----------------------------------------------------
        st.markdown("---")
        st.subheader("5. ROI画像上で2点クリックして縮尺設定")
        scale_preview = draw_points_and_lines(
            cropped_rgb,
            st.session_state.scale_points,
            close_polygon=False,
            point_color="red",
            line_color="red"
        )

        scale_clicked = streamlit_image_coordinates(
            Image.fromarray(scale_preview),
            key=f"scale_click_image_{st.session_state.scale_canvas_key}"
        )

        if scale_clicked is not None:
            current_click = (int(scale_clicked["x"]), int(scale_clicked["y"]))
            if current_click != st.session_state.scale_last_click:
                if len(st.session_state.scale_points) >= 2:
                    st.session_state.scale_points = [current_click]
                else:
                    st.session_state.scale_points.append(current_click)
                st.session_state.scale_last_click = current_click

        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("縮尺点を完全リセット"):
                st.session_state.scale_points = []
                st.session_state.scale_last_click = None
                st.session_state.scale_canvas_key += 1
                st.rerun()
        with sc2:
            if st.button("縮尺の2点目だけ取り直す"):
                if len(st.session_state.scale_points) >= 1:
                    st.session_state.scale_points = [st.session_state.scale_points[0]]
                else:
                    st.session_state.scale_points = []
                st.session_state.scale_last_click = None
                st.session_state.scale_canvas_key += 1
                st.rerun()

        if len(st.session_state.scale_points) > 0:
            st.write(f"1点目: x={st.session_state.scale_points[0][0]}, y={st.session_state.scale_points[0][1]}")
        if len(st.session_state.scale_points) > 1:
            st.write(f"2点目: x={st.session_state.scale_points[1][0]}, y={st.session_state.scale_points[1][1]}")

            pix_len = pixel_distance(st.session_state.scale_points[0], st.session_state.scale_points[1])
            meters_per_pixel = known_distance_m / pix_len if pix_len > 0 else np.nan
            pixels_per_meter = 1.0 / meters_per_pixel if meters_per_pixel > 0 else np.nan

            st.success("縮尺を計算しました。")
            s1, s2, s3 = st.columns(3)
            s1.metric("Pixel length", f"{pix_len:.2f} px")
            s2.metric("m / pixel", f"{meters_per_pixel:.6f}")
            s3.metric("pixel / m", f"{pixels_per_meter:.2f}")

            # ------------------------------------------------
            # Grid calculation
            # ------------------------------------------------
            grid_values, grid_df, grid_px_int = compute_grid_cover(
                mask=mask,
                meters_per_pixel=meters_per_pixel,
                grid_size_m=grid_size_m,
                drop_partial=drop_partial
            )

            st.markdown("---")
            st.subheader("6. グリッド被覆率")
            st.write(f"{grid_size_m:.2f} m グリッド ≒ {grid_px_int} px")

            grid_img = draw_grid_on_image(overlay, grid_px_int)

            g1, g2 = st.columns(2)
            with g1:
                st.image(grid_img, caption="グリッド重ね合わせ", use_container_width=True)
            with g2:
                fig = create_heatmap_figure(
                    grid_values,
                    title=f"{grid_size_m:.2f} m Grid Cover Heatmap (%)"
                )
                st.pyplot(fig)

            st.subheader("グリッド別データ")
            st.dataframe(grid_df, use_container_width=True)

            summary_df = pd.DataFrame([{
                "method": method,
                "roi_total_cover_percent": total_cover,
                "known_distance_m": known_distance_m,
                "clicked_pixel_length": pix_len,
                "meters_per_pixel": meters_per_pixel,
                "pixels_per_meter": pixels_per_meter,
                "grid_size_m": grid_size_m,
                "grid_size_px": grid_px_int,
                "roi_width_px": roi_w,
                "roi_height_px": roi_h,
            }])

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "summary CSV をダウンロード",
                    data=summary_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="groundcover_summary.csv",
                    mime="text/csv"
                )
            with d2:
                st.download_button(
                    "grid cover CSV をダウンロード",
                    data=grid_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="groundcover_grid_cover.csv",
                    mime="text/csv"
                )

    else:
        st.info("まず元画像上で4点クリックし、ROIを切り出してください。")

else:
    st.info("左のサイドバーから画像をアップロードしてください。")