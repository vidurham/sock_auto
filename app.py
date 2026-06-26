"""
Sock Mockup Extractor — upload PDF(s), get clean FLAT VIEW PNG, BMP, palette JSON, color column.

Run from this folder (works even when Scripts\\ is not on PATH):

  python -m streamlit run app.py

Or double-click run.bat (Windows).

Password: set APP_PASSWORD in Streamlit Cloud → Settings → Secrets, or in
`.streamlit/secrets.toml` locally (see `.streamlit/secrets.toml.example`).
"""

from __future__ import annotations

import hmac
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

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from bitmap_editor import bitmap_editor
from sock_extractor.core import process_full_pdf, resolve_output_size
from sock_extractor.product_specs import (
    DEFAULT_SPEC,
    DEFAULT_SPEC_INDEX,
    PRODUCT_SPECS,
    ProductSpec,
)

OUTPUT_PARENT = _ROOT / "output"
SESSION_KEY = "sock_last_run"
AUTH_SESSION_KEY = "sock_authenticated"
BITMAP_EDIT_KEY = "bitmap_edit_target"
REQUIRE_PASSWORD = True
LOGO_PATH = _ROOT / "assets" / "csl_logo.png"


_HIDE_STREAMLIT_STYLE = """
<style>
/* Transparent header bar; keep ⋮ menu only (do not hide header entirely) */
header[data-testid="stHeader"] {
    background: transparent;
}

footer {visibility: hidden;}

.block-container {
    padding-top: 1rem;
}

/* Streamlit Cloud: Share, star, edit, GitHub — last toolbar item is ⋮ menu */
[data-testid="stToolbar"] {
    right: 0.5rem;
}
[data-testid="stToolbar"] a,
[data-testid="stToolbar"] > div > *:not(:last-child),
[data-testid="stToolbar"] button:not(:last-of-type),
[data-testid="stToolbar"] [data-testid*="Share"],
[data-testid="stToolbar"] [data-testid*="Favorite"],
[data-testid="stToolbar"] [data-testid*="GitHub"],
[data-testid="stToolbar"] [data-testid*="Edit"],
[data-testid="stToolbar"] button[aria-label*="Share" i],
[data-testid="stToolbar"] button[aria-label*="Favorite" i],
[data-testid="stToolbar"] button[aria-label*="Star" i],
[data-testid="stToolbar"] button[aria-label*="Edit" i],
[data-testid="stToolbar"] button[aria-label*="GitHub" i] {
    display: none !important;
}
.stAppDeployButton {display: none !important;}
</style>
"""

_LOGIN_PAGE_STYLE = """
<style>
section.main {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: calc(100vh - 3rem);
}
section.main > div {
    width: 100%;
    display: flex !important;
    justify-content: center !important;
}
.main .block-container {
    width: 16.5rem !important;
    max-width: 16.5rem !important;
    min-width: 16.5rem !important;
    margin: 0 auto !important;
    padding: 0 0.5rem 1rem !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}
.main .block-container [data-testid="stHorizontalBlock"] {
    justify-content: center !important;
    width: 100% !important;
}
.main .block-container [data-testid="stVerticalBlock"],
.main .block-container [data-testid="column"] {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.main .block-container [data-testid="element-container"]:has([data-testid="stImage"]) {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}
[data-testid="stImage"] {
    width: 6.25rem !important;
    margin: 0 auto 0.35rem !important;
    display: flex !important;
    justify-content: center !important;
}
[data-testid="stImage"] > div,
[data-testid="stImage"] img {
    width: 6.25rem !important;
    max-width: 6.25rem !important;
    margin: 0 auto !important;
    display: block !important;
}
.login-heading {
    text-align: center;
    width: 100%;
    margin: 0 0 0.85rem 0;
}
.login-heading h1 {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin: 0 0 0.2rem 0;
    line-height: 1.3;
}
.login-heading p {
    color: rgba(250, 250, 250, 0.5);
    font-size: 0.72rem;
    margin: 0;
    line-height: 1.35;
}
div[data-testid="stForm"] {
    width: 16.5rem !important;
    max-width: 16.5rem !important;
    margin: 0 auto !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 0.65rem 0.7rem 0.5rem !important;
    background: rgba(255, 255, 255, 0.02) !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
}
div[data-testid="stForm"] input {
    font-size: 0.8rem !important;
    padding: 0.35rem 0.5rem !important;
    min-height: 0 !important;
    border-radius: 6px !important;
}
div[data-testid="stForm"] [data-testid="stFormSubmitButton"],
div[data-testid="stForm"] [data-testid="stFormSubmitButton"] > div {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin-top: 0.15rem;
}
div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    width: auto !important;
    min-height: 1.75rem !important;
    padding: 0.25rem 1rem !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    display: block !important;
}
.login-footer {
    text-align: center;
    width: 100%;
    margin-top: 0.65rem;
    font-size: 0.65rem;
    color: rgba(250, 250, 250, 0.35);
}
div[data-testid="stAlert"] {
    width: 16.5rem !important;
    max-width: 16.5rem !important;
    margin: 0.5rem auto 0 !important;
    font-size: 0.75rem;
    padding: 0.4rem 0.55rem;
}
</style>
"""


def _hide_streamlit_chrome() -> None:
    st.markdown(_HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)


def _app_password() -> str:
    """Password from Streamlit secrets (Cloud) or APP_PASSWORD env var (local)."""
    try:
        value = st.secrets["APP_PASSWORD"]
        if value:
            return str(value)
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    return os.environ.get("APP_PASSWORD", "")


def _render_login_page(expected: str) -> None:
    st.markdown(_LOGIN_PAGE_STYLE, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1, 1])
    with col:
        if LOGO_PATH.is_file():
            _l, logo_mid, _r = st.columns([2, 3, 2])
            with logo_mid:
                st.image(str(LOGO_PATH), width=100)
        else:
            st.markdown(
                """
                <div class="login-heading">
                  <h1>Custom Sock Lab</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="login-heading">
              <h1>Sock Mockup Extractor</h1>
              <p>Sign in to continue</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login", clear_on_submit=False, border=False):
            pwd = st.text_input(
                "Password",
                type="password",
                placeholder="Password",
                autocomplete="current-password",
                label_visibility="collapsed",
            )
            _bl, btn_mid, _br = st.columns([2, 3, 2])
            with btn_mid:
                submitted = st.form_submit_button("Sign in", type="primary")

        if submitted:
            if hmac.compare_digest(pwd, expected):
                st.session_state[AUTH_SESSION_KEY] = True
                st.rerun()
            st.error("Incorrect password. Please try again.")

        st.markdown(
            '<p class="login-footer">Team access only</p>',
            unsafe_allow_html=True,
        )


def _require_password() -> bool:
    if not REQUIRE_PASSWORD:
        return True
    if st.session_state.get(AUTH_SESSION_KEY):
        return True

    expected = _app_password()
    if not expected:
        st.error(
            "APP_PASSWORD is not configured. Add it under Streamlit Cloud "
            "→ Settings → Secrets, or set APP_PASSWORD in your environment."
        )
        return False

    _render_login_page(expected)
    return False


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


def load_bmp_indices(bmp_path: str | Path) -> np.ndarray:
    """Read a paletted production BMP into a (H,W) index array."""
    img = Image.open(bmp_path)
    if img.mode != "P":
        raise ValueError(f"Expected paletted BMP, got mode {img.mode!r}")
    return np.array(img, dtype=np.uint8)


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

        tw = info.get("target_w", target_w)
        th = info.get("target_h", target_h)
        src_note = {"manual": "manual selection",
                    "auto": "auto-detected",
                    "auto (geometry-resolved)": "auto-detected (size cross-checked against artwork)"}
        st.caption(
            f"Size: **{info.get('size_label', f'{tw}×{th} px')}**  ·  {tw}×{th}px"
            + (f"  ·  {src_note.get(info.get('size_source'), info.get('size_source'))}"
               if info.get("size_source") else "")
        )

        pal = info.get("palette", [])
        st.caption(
            f"Palette ({len(pal)} colors): "
            + ", ".join(f"rgb{c}" for c in pal)
        )

        pal_img = palette_preview_image(pal, tw, th)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**BMP (production)** · `{tw}×{th}`")
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
            st.markdown(f"**Clean FLAT VIEW** · `{tw}×{th}`")
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
                f"**Palette preview** · `{tw}×{th}` "
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
            st.image(col_bytes, width="stretch")

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

        edit_id = f"{run_id}::{base}"
        eb1, eb2 = st.columns([1, 4])
        with eb1:
            if st.button(
                "Edit bitmap",
                key=f"editbmp_{edit_id}",
                help="Open the zoomed 1px palette brush on this production BMP",
            ):
                st.session_state[BITMAP_EDIT_KEY] = edit_id
                st.rerun()
        with eb2:
            if st.session_state.get(BITMAP_EDIT_KEY) == edit_id:
                if st.button("Close editor", key=f"closebmp_{edit_id}"):
                    st.session_state[BITMAP_EDIT_KEY] = None
                    st.rerun()

        if st.session_state.get(BITMAP_EDIT_KEY) == edit_id:
            palette = [tuple(int(v) for v in c) for c in pal]
            try:
                indices = load_bmp_indices(info["bmp_path"])
            except Exception as e:
                st.error(f"Could not load BMP for editing: {e}")
            else:
                st.markdown("##### Bitmap editor")
                st.caption(
                    "Click grid cells to paint. Use **Fit all** / zoom in the toolbar, "
                    "then download the edited BMP when done."
                )
                bitmap_editor(
                    indices,
                    palette,
                    stack_key=edit_id,
                    design_name=base,
                )


def main() -> None:
    st.set_page_config(
        page_title="Sock Mockup Extractor",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )
    _hide_streamlit_chrome()
    if not _require_password():
        st.stop()

    if REQUIRE_PASSWORD:
        _, logout_col = st.columns([5, 1])
        with logout_col:
            if st.button("Log out", type="secondary", width="stretch"):
                st.session_state[AUTH_SESSION_KEY] = False
                st.rerun()

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
            "paletted BMP at the auto-detected (or chosen) product size, JSON palette, and color-column preview. "
            "Results stay on screen after each download."
        )

    AUTO_SIZE = "__auto__"
    size_choice = st.selectbox(
        "Output size",
        options=[AUTO_SIZE, *PRODUCT_SPECS],
        index=0,
        format_func=lambda s: "Auto-detect from each PDF (recommended)" if s == AUTO_SIZE else s.label,
        help=(
            "Auto-detect reads the product type, length and flat-view proportions "
            "from each PDF and picks the matching production size. Choose a specific "
            "size to force it for every file."
        ),
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
            width="stretch",
        )
    with btn_right:
        if st.session_state[SESSION_KEY] is not None:
            if st.button("Clear results from page"):
                st.session_state[SESSION_KEY] = None
                st.session_state[BITMAP_EDIT_KEY] = None
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
                        if size_choice == AUTO_SIZE:
                            r = resolve_output_size(str(pdf_path))
                            tw, th = r["width"], r["height"]
                            size_label = (
                                r["spec"].label if r["spec"]
                                else f"Derived from artwork · {tw}×{th} px"
                            )
                            size_source = r["source"]
                        else:
                            tw, th = size_choice.width, size_choice.height
                            size_label = size_choice.label
                            size_source = "manual"
                        info = process_full_pdf(
                            str(pdf_path),
                            str(job_dir),
                            target_w=tw,
                            target_h=th,
                        )
                        info["_basename"] = stem
                        info["target_w"], info["target_h"] = tw, th
                        info["size_label"], info["size_source"] = size_label, size_source
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
                    "target_w": DEFAULT_SPEC.width,
                    "target_h": DEFAULT_SPEC.height,
                    "product_label": (
                        "Auto-detected per file" if size_choice == AUTO_SIZE
                        else size_choice.label
                    ),
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
