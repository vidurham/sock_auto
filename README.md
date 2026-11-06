# Sock Mockup → Bitmap

Turns a Custom Sock Lab mockup (PDF, or a PNG/JPG screenshot of one) into the
production bitmap: 168 px wide at the right height for the product, swatch
colours only, cuff / toe / heel guides removed.

## Setup (once)

Needs Python 3.10+.

```bash
pip install -r requirements.txt
```

No password is needed to run it locally. To add a login (as the hosted version
has), copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set
`APP_PASSWORD`, or set an `APP_PASSWORD` environment variable.

## Run the app

Windows: double-click `run.bat`. Anywhere else:

```bash
python -m streamlit run app.py
```

Open the address it prints (usually http://localhost:8501). Then:

1. Leave **Output size** on Auto-detect, or pick a size to force it.
2. Drop in PDFs or images and click **Process PDFs**.
3. Download each BMP, or everything as a ZIP.

**Edit bitmap** opens a pixel editor for touch-ups. **Pixel-art stars** is
optional: it draws small round dots as plus/diamond shapes.

## Convert a whole folder (terminal)

```bash
python batch.py path/to/mockups/
```

Results go to `output/batch/<name>/` with a `summary.csv`. Options:

- `-o out/` chooses the output folder.
- `--size "Athletic - Cotton & Wool/Crew"` forces a size.
- `--pixel-stars` turns on the pixel-art stars.

## Accuracy check

```bash
python tools/score_pdf.py
```

This compares output against the designers' bitmaps in `finished_bit/` and
writes side-by-side images to `output/score_pdf/`. See `RULES.md` for every
rule, experiment and result. (`tools/learn.py` additionally needs
`pip install scikit-learn`; nothing else does.)

## Make a zip to share

```bash
git archive --format=zip -o sock_auto.zip HEAD
```

This includes only tracked files: no `output/`, no `.git` history, and no
`secrets.toml`.
