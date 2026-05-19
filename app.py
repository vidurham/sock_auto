"""
Sock Mockup Extractor — upload PDF(s), get clean FLAT VIEW PNG, BMP, palette JSON, color column.

Run from this folder (works even when Scripts\\ is not on PATH):

  python -m streamlit run app.py

Or double-click run.bat (Windows).
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path

# Allow `streamlit run path/to/app.py` from any working directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from PIL import Image, ImageDraw

from sock_extractor.core import process_full_pdf
from sock_extractor.product_specs import (
    DEFAULT_SPEC,
    DEFAULT_SPEC_INDEX,
    PRODUCT_SPECS,
    ProductSpec,
)

OUTPUT_PARENT = _ROOT / "output"
SESSION_KEY = "sock_last_run"
LOGO_PATH = _ROOT / "assets" / "csl_logo.png"


def _hide_streamlit_chrome() -> None:
    """Hide Streamlit Cloud GitHub / Edit toolbar actions and local Deploy button."""
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"] a[href*="github.com"],
        [data-testid="stToolbar"] a[href*="share.streamlit.io"],
        [data-testid="stToolbar"] button[title*="Edit"],
        [data-testid="stToolbar"] button[aria-label*="Edit"],
        [data-testid="stToolbar"] button[title*="GitHub"],
        [data-testid="stToolbar"] button[aria-label*="GitHub"] {
            display: none !important;
        }
        .stAppDeployButton {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def palette_preview_image(
    palette: list[tuple[int, int, int]],
    width: int,
    height: int,
) -> Image.Image:
    """Same pixel size as BMP / design PNG: horizontal bands, one per swatch."""
    if not palette:
        out = Image.new("RGB", (width, height), (245, 245, 245))
        return out
    n = len(palette)
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for i, rgb in enumerate(palette):
        y0 = int(round(i * height / n))
        y1 = int(round((i + 1) * height / n))
        if y1 <= y0:
            y1 = y0 + 1
        draw.rectangle([0, y0, width, y1], fill=rgb)
        if i < n - 1:
            draw.line([(0, y1), (width, y1)], fill=(255, 255, 255), width=1)
    return img


def zip_run_folder(run_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(run_dir.rglob("*")):
            if fp.is_file():
                arc = fp.relative_to(run_dir).as_posix()
                zf.write(fp, arcname=arc)
    buf.seek(0)
    return buf.getvalue()


def render_results(
    run_id: str,
    run_dir: Path,
    results: list[dict],
    errors: list[tuple[str, str]],
    target_w: int,
    target_h: int,
) -> None:
    """Show ZIP, previews, and downloads — reads from disk so reruns (e.g. after a download) still work."""
    if errors:
        for name, msg in errors:
            st.error(f"**{name}:** {msg}")

    if not results:
        st.warning("No files processed successfully in this run.")
        return

    zip_bytes = zip_run_folder(run_dir)
    st.success(
        f"Processed {len(results)} file(s). Files are saved under "
        f"`{run_dir.relative_to(_ROOT)}` — downloads below do not clear this."
    )
    st.download_button(
        label="Download all (ZIP)",
        data=zip_bytes,
        file_name=f"sock_extractor_{run_id}.zip",
        mime="application/zip",
        key=f"zip_dl_{run_id}",
    )

    preview_w = 220
    for info in results:
        base = info["_basename"]
        st.divider()
        st.subheader(base)

        pal = info.get("palette", [])
        st.caption(
            f"Palette ({len(pal)} colors): "
            + ", ".join(f"rgb{c}" for c in pal)
        )

        pal_img = palette_preview_image(pal, target_w, target_h)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**BMP (production)** · `{target_w}×{target_h}`")
            with open(info["bmp_path"], "rb") as f:
                bmp_data = f.read()
            st.download_button(
                label=f"{base}.bmp",
                data=bmp_data,
                file_name=os.path.basename(info["bmp_path"]),
                mime="image/bmp",
                key=f"bmp_{run_id}_{base}",
            )
            st.image(info["bmp_path"], width=preview_w)

        with c2:
            st.markdown(f"**Clean FLAT VIEW** · `{target_w}×{target_h}`")
            with open(info["design_path"], "rb") as f:
                png_data = f.read()
            st.download_button(
                label=f"{base}_design.png",
                data=png_data,
                file_name=os.path.basename(info["design_path"]),
                mime="image/png",
                key=f"png_{run_id}_{base}",
            )
            st.image(info["design_path"], width=preview_w)

        with c3:
            st.markdown(
                f"**Palette preview** · `{target_w}×{target_h}` "
                "(same size as BMP — swatches top→bottom)"
            )
            buf = io.BytesIO()
            pal_img.save(buf, format="PNG")
            pprev = buf.getvalue()
            st.download_button(
                label=f"{base}_palette_preview.png",
                data=pprev,
                file_name=f"{base}_palette_preview.png",
                mime="image/png",
                key=f"pprev_{run_id}_{base}",
            )
            st.image(pal_img, width=preview_w)

        with st.expander("PDF color column (cropped from template)", expanded=False):
            st.caption(
                "Raw crop from the mockup PDF — useful for checking labels; "
                "not the same dimensions as the BMP."
            )
            with open(info["colors_path"], "rb") as f:
                col_bytes = f.read()
            st.download_button(
                label=f"{base}_colors.png",
                data=col_bytes,
                file_name=os.path.basename(info["colors_path"]),
                mime="image/png",
                key=f"col_{run_id}_{base}",
            )
            st.image(col_bytes, use_container_width=True)

        if info.get("palette_path") and os.path.isfile(info["palette_path"]):
            with open(info["palette_path"], "rb") as f:
                json_data = f.read()
            st.download_button(
                label=f"{base}_palette.json",
                data=json_data,
                file_name=os.path.basename(info["palette_path"]),
                mime="application/json",
                key=f"json_{run_id}_{base}",
            )


def main() -> None:
    st.set_page_config(page_title="Sock Mockup Extractor", layout="wide")
    _hide_streamlit_chrome()

    logo_col, title_col = st.columns([1, 2.5], gap="large")
    with logo_col:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=260)
        else:
            st.caption("Logo not found: `assets/csl_logo.png`")
    with title_col:
        st.title("Sock Mockup Extractor")
        st.caption(
            "Custom Sock Lab–style PDFs: extracts FLAT VIEW (heel guides removed), "
            "paletted BMP at the selected product size, JSON palette, and color-column preview. "
            "Results stay on screen after each download."
        )

    spec: ProductSpec = st.selectbox(
        "Product type & style",
        options=PRODUCT_SPECS,
        index=DEFAULT_SPEC_INDEX,
        format_func=lambda s: s.label,
        help=f"Output size for BMP and design PNG. Default: {DEFAULT_SPEC.label}",
    )

    uploaded = st.file_uploader(
        "Drop PDF(s) here or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = None

    btn_left, btn_right = st.columns([1, 3])
    with btn_left:
        process_clicked = st.button(
            "Process PDFs",
            type="primary",
            disabled=not uploaded,
            use_container_width=True,
        )
    with btn_right:
        if st.session_state[SESSION_KEY] is not None:
            if st.button("Clear results from page"):
                st.session_state[SESSION_KEY] = None
                st.rerun()

    spin_left, _ = st.columns([1, 3])
    with spin_left:
        if process_clicked and uploaded:
            with st.spinner("Processing…"):
                run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
                run_dir = OUTPUT_PARENT / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                results: list[dict] = []
                errors: list[tuple[str, str]] = []

                for i, uf in enumerate(uploaded):
                    orig_name = Path(uf.name).name
                    safe_name = "".join(
                        c for c in orig_name if c.isascii() and (c.isalnum() or c in "._- ")
                    )
                    if not safe_name.lower().endswith(".pdf"):
                        safe_name = Path(orig_name).stem + ".pdf"
                    stem = Path(safe_name).stem
                    job_dir = run_dir / f"{i:02d}_{stem}"
                    job_dir.mkdir(parents=True, exist_ok=True)
                    pdf_path = job_dir / safe_name
                    with open(pdf_path, "wb") as f:
                        f.write(uf.getbuffer())

                    try:
                        info = process_full_pdf(
                            str(pdf_path),
                            str(job_dir),
                            target_w=spec.width,
                            target_h=spec.height,
                        )
                        info["_basename"] = stem
                        results.append(info)
                    except Exception as e:
                        errors.append((safe_name, str(e)))
                        try:
                            shutil.rmtree(job_dir, ignore_errors=True)
                        except OSError:
                            pass

                st.session_state[SESSION_KEY] = {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "results": results,
                    "errors": errors,
                    "target_w": spec.width,
                    "target_h": spec.height,
                    "product_label": spec.label,
                }

    if uploaded:
        st.caption(f"{len(uploaded)} PDF(s) selected — click **Process PDFs** to run.")
    elif st.session_state[SESSION_KEY] is None:
        st.info("Upload one or more PDF mockups to process.")
        return

    run = st.session_state[SESSION_KEY]
    if run is not None:
        st.markdown(
            f"**Saved output folder:** `{Path(run['run_dir']).relative_to(_ROOT)}` "
            "(you can copy files from disk anytime)"
        )
        st.caption(f"Product: {run.get('product_label', DEFAULT_SPEC.label)}")
        render_results(
            run["run_id"],
            Path(run["run_dir"]),
            run["results"],
            run["errors"],
            run.get("target_w", DEFAULT_SPEC.width),
            run.get("target_h", DEFAULT_SPEC.height),
        )


if __name__ == "__main__":
    main()
