"""
Gradio demo: ColPali (visual, multivector) vs text-chunking (OCR + nomic)
retrieval over the same ViDoRe pages, side by side.

Query/visualization layer ONLY. No indexing, OCR, or training at runtime; the
two collections are built beforehand by 3-ingest_vidore.py and
4-ingest_text_baseline.py. The retrieval code here mirrors those scripts exactly.

Prerequisites running before launch:
  - Qdrant at http://localhost:6333 with collections `pdf_pages` (ColPali
    multivector, point id = page index, payload {"row": i}) and `pdf_text`
    (single-vector chunks, payload {"page": i, "text": ...}).
  - Both collections built from the SAME source (ViDoRe subset or an uploaded
    PDF) so point ids align with page indices.

Collections are (re)built from the Ingestion tab — either a ViDoRe subset (with
gold queries for HIT/MISS) or an uploaded PDF (free-text search only).

The text query is embedded in-process with nomic-embed-text (transformers), so
no Ollama server is needed at app runtime. The `vidore_text` collection must have
been indexed with the matching nomic model + `search_document:` prefix.

    python app.py
"""

import argparse
import json
import os
import time

import fitz  # PyMuPDF — render uploaded PDF pages to images
import gradio as gr
import pytesseract
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageOps
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer

from colpali_engine.models import ColPali, ColPaliProcessor

# --------------------------------------------------------------------------- #
# Config (defaults match the ingest scripts; override via CLI). Must match what
# was indexed so point ids line up with dataset row indices.
# --------------------------------------------------------------------------- #


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--query-col", default="query")
    ap.add_argument("--model", default="vidore/colpali-v1.3")
    ap.add_argument("--pages-collection", default="pdf_pages")
    ap.add_argument("--text-collection", default="pdf_text")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    # In-process query embedder. Must be the same nomic model the baseline script
    # indexed `vidore_text` with (Ollama's nomic-embed-text == this HF checkpoint).
    ap.add_argument("--text-model", default="nomic-ai/nomic-embed-text-v1.5")
    # nomic uses task prefixes; queries are embedded with the search_query prefix
    # (docs were indexed with search_document in the baseline script).
    ap.add_argument("--query-prefix", default="search_query: ")
    # Ingestion-side knobs (mirror 3-ingest_vidore.py / 4-ingest_text_baseline.py).
    ap.add_argument("--doc-prefix", default="search_document: ",
                    help="nomic prefix for indexed chunks (must match what queries search)")
    ap.add_argument("--colpali-batch", type=int, default=4, help="image embed batch size")
    ap.add_argument("--embed-batch", type=int, default=32, help="chunk embed batch size")
    ap.add_argument("--chunk-size", type=int, default=200, help="words per text chunk")
    ap.add_argument("--chunk-overlap", type=int, default=40, help="word overlap between chunks")
    ap.add_argument("--tesseract-cmd", default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    help="path to tesseract.exe for the text-ingestion OCR step")
    ap.add_argument("--candidate-pool", type=int, default=100,
                    help="text chunks to retrieve before collapsing to pages")
    ap.add_argument("--top-n", type=int, default=3, help="default images per side")
    ap.add_argument("--max-top-n", type=int, default=5)
    ap.add_argument("--metrics-file", default="metrics.json")
    ap.add_argument("--state-dir", default="app_state",
                    help="dir holding the last-source state and cached PDF page images")
    ap.add_argument("--server-name", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    return ap.parse_args()


ARGS = parse_args()

# Default aggregate metrics (DocVQA); overridden by metrics.json if present.
DEFAULT_METRICS = {
    "label": "DocVQA",
    "colpali_hit5": 0.80,
    "colpali_mrr5": 0.715,
    "text_hit5": 0.48,
    "text_mrr5": 0.360,
}

GOLD_COLOR = "#22c55e"  # green border for the gold page

# --------------------------------------------------------------------------- #
# Startup singletons — model, processor, Qdrant client, dataset images/queries.
# Built ONCE at import; reused for every query (requirements 6.4).
# --------------------------------------------------------------------------- #


def pick_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32  # slow but works for a small demo


def load_metrics(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {**DEFAULT_METRICS, **json.load(f)}
        except Exception as e:  # noqa: BLE001 - non-fatal, fall back to defaults
            print(f"[metrics] could not read {path}: {e}; using defaults")
    return DEFAULT_METRICS


# --- Persistent state: remember the last ingested source across restarts -----
STATE_FILE = os.path.join(ARGS.state_dir, "state.json")
PDF_PAGES_DIR = os.path.join(ARGS.state_dir, "pages")


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - no/invalid state -> fall back to defaults
        return None


def save_state(state):
    os.makedirs(ARGS.state_dir, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _dataset_images_queries(dataset, split, limit):
    ds = load_dataset(dataset, split=split)
    if limit:
        ds = ds.select(range(min(int(limit), len(ds))))
    if ARGS.image_col not in ds.column_names or ARGS.query_col not in ds.column_names:
        raise SystemExit(
            f"Expected columns '{ARGS.image_col}' and '{ARGS.query_col}'. "
            f"Found: {ds.column_names}. Re-run with --image-col / --query-col."
        )
    imgs = [row[ARGS.image_col].convert("RGB") for row in ds]
    qs = [(i, row[ARGS.query_col]) for i, row in enumerate(ds) if row[ARGS.query_col]]
    return imgs, qs


def save_pdf_pages(images):
    """Persist rendered PDF page images so a PDF index survives an app restart."""
    os.makedirs(PDF_PAGES_DIR, exist_ok=True)
    for f in os.listdir(PDF_PAGES_DIR):
        os.remove(os.path.join(PDF_PAGES_DIR, f))
    for i, im in enumerate(images):
        im.save(os.path.join(PDF_PAGES_DIR, f"{i}.png"))


def _load_pdf_pages():
    if not os.path.isdir(PDF_PAGES_DIR):
        return []
    files = sorted(
        (f for f in os.listdir(PDF_PAGES_DIR) if f.endswith(".png")),
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    return [Image.open(os.path.join(PDF_PAGES_DIR, f)).convert("RGB") for f in files]


def load_initial_cache():
    """Restore page images/queries from the last-used source, else the default
    dataset. Returns (images, queries, source_label)."""
    st = load_state()
    if st and st.get("source") == "pdf":
        imgs = _load_pdf_pages()
        if imgs:
            print(f"Restoring last source: PDF ({len(imgs)} cached pages).")
            return imgs, [], "pdf"
        print("[state] last source was PDF but no cached pages found; using dataset.")
    dataset = (st or {}).get("dataset", ARGS.dataset)
    split = (st or {}).get("split", ARGS.split)
    limit = (st or {}).get("limit", ARGS.limit)
    print(f"Loading dataset {dataset} [{split}] (limit {limit}) ...")
    imgs, qs = _dataset_images_queries(dataset, split, limit)
    return imgs, qs, "dataset"


print("=" * 70)
DEVICE, DTYPE = pick_device_and_dtype()
print(f"Device: {DEVICE} | dtype: {DTYPE}")

print(f"Loading model {ARGS.model} ...")
# device_map loads weights straight onto the target device. Chaining .to(device)
# fails with "Cannot copy out of meta tensor" on recent transformers.
MODEL = ColPali.from_pretrained(
    ARGS.model, torch_dtype=DTYPE, device_map=DEVICE
).eval()
PROCESSOR = ColPaliProcessor.from_pretrained(ARGS.model)

print(f"Loading text embedder {ARGS.text_model} ...")
# Small (~137M) nomic model loaded in-process so query embedding is a direct
# forward pass (no Ollama HTTP round-trip). float32 on whatever device ColPali used.
TEXT_TOKENIZER = AutoTokenizer.from_pretrained(ARGS.text_model, trust_remote_code=True)
TEXT_MODEL = AutoModel.from_pretrained(
    ARGS.text_model, trust_remote_code=True
).to(DEVICE).eval()

# images[point_id] -> the retrieved page image (RGB, for drawing gold borders).
# QUERIES holds (row_id, query_text) only for a dataset source (gold pages known);
# for an uploaded PDF it is empty, which disables the dataset-specific UI.
IMAGES, QUERIES, ACTIVE_SOURCE = load_initial_cache()
print(f"Cached {len(IMAGES)} page images; {len(QUERIES)} dataset queries "
      f"(source: {ACTIVE_SOURCE}).")

CLIENT = QdrantClient(ARGS.qdrant_url, timeout=120)
for _coll in (ARGS.pages_collection, ARGS.text_collection):
    try:
        if not CLIENT.collection_exists(_coll):
            print(f"[warn] Qdrant collection '{_coll}' not found at {ARGS.qdrant_url}. "
                  "Retrieval on that side will return nothing until it is indexed.")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not reach Qdrant at {ARGS.qdrant_url}: {e}")
        break

METRICS = load_metrics(ARGS.metrics_file)

# dropdown label -> (row_id, query_text), so we can recover the gold row.
QUERY_LABELS = {f"row {rid}: {text}": (rid, text) for rid, text in QUERIES}

# --------------------------------------------------------------------------- #
# Retrieval — mirrors 3-ingest_vidore.py (ColPali) and 4-ingest_text_baseline.py
# (text). Each returns a list of (page_id, score) in rank order.
# --------------------------------------------------------------------------- #


def retrieve_colpali(query, top_n):
    try:
        batch = PROCESSOR.process_queries([query])
        batch = {k: v.to(MODEL.device) for k, v in batch.items()}
        with torch.no_grad():
            q_emb = MODEL(**batch)
        vec = q_emb[0].cpu().float().tolist()
        points = CLIENT.query_points(
            ARGS.pages_collection, query=vec, limit=top_n
        ).points
        return [(p.id, p.score) for p in points]
    except Exception as e:  # noqa: BLE001 - never crash the UI
        print(f"[colpali] retrieval failed: {e}")
        return []


def embed_texts_local(texts, prefix, batch_size=32):
    """Embed strings in-process with nomic: prefix -> forward -> mean-pool ->
    L2-normalize. One code path for both query and document (chunk) embedding."""
    out = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start : start + batch_size]]
        enc = TEXT_TOKENIZER(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            tokens = TEXT_MODEL(**enc).last_hidden_state  # (B, seq, dim)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        out.extend(pooled.cpu().float().tolist())
    return out


def embed_query_local(query):
    """Embed a single search query (search_query prefix)."""
    return embed_texts_local([query], ARGS.query_prefix, batch_size=1)[0]


def retrieve_text(query, top_n):
    try:
        vec = embed_query_local(query)
        points = CLIENT.query_points(
            ARGS.text_collection, query=vec, limit=ARGS.candidate_pool
        ).points
        # Collapse chunks to distinct pages, preserving rank order; page score =
        # the best (first-seen) chunk score. Stop once we have top_n pages.
        seen, out = set(), []
        for p in points:
            page = p.payload["page"]
            if page not in seen:
                seen.add(page)
                out.append((page, p.score))
                if len(out) >= top_n:
                    break
        return out
    except Exception as e:  # noqa: BLE001 - never crash the UI
        print(f"[text] retrieval failed: {e}")
        return []


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #


def _gold_bordered(img):
    return ImageOps.expand(img, border=10, fill=GOLD_COLOR)


def build_gallery(results, gold_row):
    """Return (gallery items [(image, caption)], hit: bool)."""
    items, hit = [], False
    for page_id, score in results:
        if page_id < 0 or page_id >= len(IMAGES):
            continue  # no cached image for this id (shouldn't happen) -> skip
        is_gold = gold_row is not None and page_id == gold_row
        img = _gold_bordered(IMAGES[page_id]) if is_gold else IMAGES[page_id]
        caption = f"r{page_id}   {score:.3f}" + ("   ★ GOLD" if is_gold else "")
        items.append((img, caption))
        hit = hit or is_gold
    return items, hit


_BADGE_STYLE = "display:inline-block;padding:4px 14px;border-radius:6px;font-weight:700;"
_BADGE_PLACEHOLDER = f"<div style='{_BADGE_STYLE}visibility:hidden;'>HIT</div>"
_ROW_STYLE = "display:flex;align-items:center;gap:12px;min-height:28px;"

# Status row reserves a fixed height, so the loading overlay is never clipped and
# the layout doesn't jump between the empty / timing / badge states.
STATUS_PLACEHOLDER = (
    f"<div style='{_ROW_STYLE}'><span style='visibility:hidden;'>⏱ 0.000 s</span></div>"
)


def _badge(gold_row, hit):
    if gold_row is None:
        return _BADGE_PLACEHOLDER  # free-typed query: no gold, keep the space
    color, label = ("#16a34a", "HIT") if hit else ("#dc2626", "MISS")
    return f"<div style='{_BADGE_STYLE}background:{color};color:white;'>{label}</div>"


def status_html(gold_row, hit, seconds):
    """One column's status row: measured retrieval time + HIT/MISS badge."""
    timer = (
        "<span style='font-variant-numeric:tabular-nums;opacity:0.85;"
        f"font-weight:600;'>⏱ {seconds:.3f} s</span>"
    )
    return f"<div style='{_ROW_STYLE}'>{timer}{_badge(gold_row, hit)}</div>"


def resolve_gold(query, dropdown_label):
    """Gold row is known only when the box still holds the picked dataset query."""
    if dropdown_label and dropdown_label in QUERY_LABELS:
        rid, text = QUERY_LABELS[dropdown_label]
        if query.strip() == text.strip():
            return rid
    return None


# Each column is its own Gradio event so it renders the instant its own search
# finishes — ColPali (fast) shows its pages while text chunking is still running,
# instead of both waiting on the slower half.


def run_colpali(query, top_n, dropdown_label):
    query = (query or "").strip()
    if not query:
        return [], STATUS_PLACEHOLDER
    gold_row = resolve_gold(query, dropdown_label)
    t0 = time.perf_counter()
    items, hit = build_gallery(retrieve_colpali(query, int(top_n)), gold_row)
    return items, status_html(gold_row, hit, time.perf_counter() - t0)


def run_text(query, top_n, dropdown_label):
    query = (query or "").strip()
    if not query:
        return [], STATUS_PLACEHOLDER
    gold_row = resolve_gold(query, dropdown_label)
    t0 = time.perf_counter()
    items, hit = build_gallery(retrieve_text(query, int(top_n)), gold_row)
    return items, status_html(gold_row, hit, time.perf_counter() - t0)


def on_pick_query(dropdown_label):
    """Fill the textbox with the selected dataset query."""
    if dropdown_label and dropdown_label in QUERY_LABELS:
        return QUERY_LABELS[dropdown_label][1]
    return gr.update()


# --------------------------------------------------------------------------- #
# Ingestion — rebuild the Qdrant collections from the GUI, reusing the already
# loaded models. Mirrors 3-ingest_vidore.py / 4-ingest_text_baseline.py exactly
# (full rebuild: delete + recreate). Streams a progress log.
# --------------------------------------------------------------------------- #


def chunk_text(text, size, overlap):
    """Sliding-window word chunks (same as the text baseline script)."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def render_pdf_pages(path, dpi=180):
    """Render every page of a PDF to a PIL image (no external poppler needed)."""
    doc = fitz.open(path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    finally:
        doc.close()
    return images


def run_ingestion(source, which, limit, pdf_path, chunk_size, chunk_overlap, tess_cmd,
                  progress=gr.Progress()):
    """Generator: (re)build the selected collection(s) from either the ViDoRe
    subset or an uploaded PDF, streaming a log. Refreshes the in-memory caches so
    the Search tab reflects the new index."""
    global IMAGES, QUERIES, QUERY_LABELS, ACTIVE_SOURCE
    lines = []

    def log(msg):
        lines.append(msg)
        return "\n".join(lines)

    t_start = time.perf_counter()
    limit = int(limit)
    do_colpali = which in ("ColPali pages", "Both")
    do_text = which in ("Text chunks", "Both")

    # Build the page images (and gold queries, if any) from the chosen source.
    if source == "Upload PDF":
        if not pdf_path:
            yield log("No PDF uploaded. Choose a file and run again.")
            return
        yield log(f"Rendering pages from {os.path.basename(pdf_path)} ...")
        raw_images = render_pdf_pages(pdf_path)
        queries = []  # an arbitrary PDF has no gold queries -> free-text search only
        yield log(f"  {len(raw_images)} pages (no gold queries; HIT/MISS disabled).")
    else:
        yield log(f"Loading {ARGS.dataset} [{ARGS.split}] (limit {limit}) ...")
        ds = load_dataset(ARGS.dataset, split=ARGS.split)
        ds = ds.select(range(min(limit, len(ds))))
        raw_images = [row[ARGS.image_col] for row in ds]
        queries = [(i, row[ARGS.query_col]) for i, row in enumerate(ds) if row[ARGS.query_col]]
        yield log(f"  {len(raw_images)} pages, {len(queries)} queries.")

    if not raw_images:
        yield log("No pages to index. Aborting.")
        return
    display_images = [im.convert("RGB") for im in raw_images]

    # ---- ColPali pages (multivector, MaxSim) ----
    if do_colpali:
        yield log(f"\n[ColPali] embedding {len(raw_images)} pages on {DEVICE} ...")
        embs = []
        bs = ARGS.colpali_batch
        for start in progress.tqdm(range(0, len(raw_images), bs), desc="ColPali embed"):
            batch_imgs = raw_images[start : start + bs]
            batch = PROCESSOR.process_images(batch_imgs)
            batch = {k: v.to(MODEL.device) for k, v in batch.items()}
            with torch.no_grad():
                emb = MODEL(**batch)
            embs.extend(list(torch.unbind(emb.to("cpu").float())))
        dim = embs[0].shape[-1]
        yield log(f"[ColPali] (re)creating '{ARGS.pages_collection}' (dim={dim}, MaxSim) ...")
        if CLIENT.collection_exists(ARGS.pages_collection):
            CLIENT.delete_collection(ARGS.pages_collection)
        CLIENT.create_collection(
            ARGS.pages_collection,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        # Small upsert batches: multivector points are large and overflow Qdrant
        # if pushed all at once (WinError 10053).
        for start in range(0, len(embs), 8):
            chunk = embs[start : start + 8]
            CLIENT.upsert(
                ARGS.pages_collection,
                points=[
                    models.PointStruct(id=start + j, vector=e.tolist(), payload={"row": start + j})
                    for j, e in enumerate(chunk)
                ],
                wait=True,
            )
        yield log(f"[ColPali] indexed {len(embs)} pages.")

    # ---- Text chunks (OCR -> chunk -> nomic single-vector) ----
    if do_text:
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
        yield log(f"\n[Text] OCR + chunk {len(raw_images)} pages ...")
        chunk_texts, chunk_pages, blank = [], [], 0
        for i, im in enumerate(progress.tqdm(raw_images, desc="OCR")):
            text = pytesseract.image_to_string(im)
            pcs = chunk_text(text, int(chunk_size), int(chunk_overlap))
            if not pcs:
                blank += 1
            for c in pcs:
                chunk_texts.append(c)
                chunk_pages.append(i)
        yield log(f"[Text] {len(chunk_texts)} chunks; {blank} pages yielded no OCR text.")
        if not chunk_texts:
            yield log("[Text] OCR produced no text — check the Tesseract path. Skipped.")
        else:
            yield log(f"[Text] embedding {len(chunk_texts)} chunks (nomic '{ARGS.doc_prefix.strip()}') ...")
            vecs = embed_texts_local(chunk_texts, ARGS.doc_prefix, batch_size=ARGS.embed_batch)
            dim = len(vecs[0])
            yield log(f"[Text] (re)creating '{ARGS.text_collection}' (dim={dim}) ...")
            if CLIENT.collection_exists(ARGS.text_collection):
                CLIENT.delete_collection(ARGS.text_collection)
            CLIENT.create_collection(
                ARGS.text_collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            for start in range(0, len(vecs), 256):
                end = start + 256
                CLIENT.upsert(
                    ARGS.text_collection,
                    points=[
                        models.PointStruct(
                            id=idx, vector=v,
                            payload={"page": chunk_pages[idx], "text": chunk_texts[idx][:300]},
                        )
                        for idx, v in zip(range(start, end), vecs[start:end])
                    ],
                    wait=True,
                )
            yield log(f"[Text] indexed {len(vecs)} chunks.")

    # Refresh in-memory caches so the Search tab uses the freshly built index.
    IMAGES = display_images
    QUERIES = queries
    QUERY_LABELS = {f"row {rid}: {text}": (rid, text) for rid, text in QUERIES}

    # Persist the source so the next launch restores it (and the right UI state).
    if source == "Upload PDF":
        ACTIVE_SOURCE = "pdf"
        save_pdf_pages(display_images)
        save_state({"source": "pdf", "pdf": os.path.basename(pdf_path), "n_pages": len(display_images)})
    else:
        ACTIVE_SOURCE = "dataset"
        save_state({"source": "dataset", "dataset": ARGS.dataset,
                    "split": ARGS.split, "limit": limit})
    yield log(f"\nDone in {time.perf_counter() - t_start:.1f}s. Search tab now reflects the new index.")


def after_ingest():
    """Sync the dataset-specific Search UI to the new source: the query dropdown
    and the aggregate-metrics panel are shown only when gold queries exist."""
    has_queries = bool(QUERIES)
    return (
        gr.update(choices=list(QUERY_LABELS.keys()), value=None, visible=has_queries),
        gr.update(visible=has_queries),
    )


# --------------------------------------------------------------------------- #
# Setup — Qdrant connection status and collection initialization
# --------------------------------------------------------------------------- #

MULTIVECTOR_LABEL = "Multivector (ColPali, MaxSim)"
SINGLEVECTOR_LABEL = "Single-vector (text)"


def _status_html(state, msg, sub=""):
    color = {"ok": "#16a34a", "err": "#dc2626", "unknown": "#6b7280"}[state]
    dot = (f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
           f"background:{color};margin-right:8px;flex:none;'></span>")
    sub_html = f"<div style='opacity:0.75;margin-top:4px;'>{sub}</div>" if sub else ""
    return (f"<div style='min-height:24px;'><div style='display:flex;align-items:center;'>"
            f"{dot}<span>{msg}</span></div>{sub_html}</div>")


def qdrant_status():
    """Ping Qdrant and report connection state + existing collections (with counts)."""
    try:
        colls = [c.name for c in CLIENT.get_collections().collections]
        bits = []
        for n in colls:
            try:
                bits.append(f"{n} ({CLIENT.count(n, exact=True).count})")
            except Exception:  # noqa: BLE001
                bits.append(n)
        detail = "Collections: " + (", ".join(bits) if bits else "none")
        return _status_html("ok", f"Connected to {ARGS.qdrant_url}", detail)
    except Exception as e:  # noqa: BLE001
        return _status_html("err", f"Cannot reach Qdrant at {ARGS.qdrant_url}", str(e))


def init_collection(name, vtype, size, recreate):
    """Create a collection (type/size configurable)."""
    name = (name or "").strip()
    if not name:
        return _status_html("err", "Provide a collection name."), qdrant_status()
    try:
        size = int(size)
        if CLIENT.collection_exists(name):
            if not recreate:
                return (
                    _status_html("err", f"Collection '{name}' already exists.",
                                 "Tick 'Recreate' to delete and replace it."),
                    qdrant_status(),
                )
            CLIENT.delete_collection(name)
        if vtype == MULTIVECTOR_LABEL:
            vectors_config = models.VectorParams(
                size=size, distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
        else:
            vectors_config = models.VectorParams(size=size, distance=models.Distance.COSINE)
        CLIENT.create_collection(name, vectors_config=vectors_config)
        return (
            _status_html("ok", f"Created '{name}'", f"{vtype}, size {size}, cosine"),
            qdrant_status(),
        )
    except Exception as e:  # noqa: BLE001
        return _status_html("err", f"Failed to create '{name}'", str(e)), qdrant_status()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

METRICS_LINE = (
    f"**Aggregate ({METRICS['label']}):**  "
    f"ColPali hit@5 {METRICS['colpali_hit5']:.2f} / MRR {METRICS['colpali_mrr5']:.3f}"
    f"  &nbsp;|&nbsp;  "
    f"text hit@5 {METRICS['text_hit5']:.2f} / MRR {METRICS['text_mrr5']:.3f}"
    "  _(precomputed, not live)_"
)

with gr.Blocks(title="ColPali vs Text-Chunking Retrieval") as demo:
    gr.Markdown("# ColPali (visual) vs Text-Chunking (OCR + nomic) — page retrieval")

    with gr.Tabs():
        with gr.Tab("Search"):
            with gr.Row():
                query_box = gr.Textbox(
                    label="Query",
                    placeholder="Type anything, or pick a dataset query below",
                    scale=4, autofocus=True,
                )
                retrieve_btn = gr.Button("Retrieve", variant="primary", scale=1)

            with gr.Row():
                query_dropdown = gr.Dropdown(
                    label="Dataset queries (gold page known)",
                    choices=list(QUERY_LABELS.keys()),
                    value=None, scale=4, filterable=True,
                    visible=bool(QUERIES),  # hidden for an uploaded PDF (no gold)
                )
                top_n_slider = gr.Slider(
                    label="Top-N images per side", minimum=1, maximum=ARGS.max_top_n,
                    value=ARGS.top_n, step=1, scale=1,
                )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ColPali (visual)")
                    cp_badge = gr.HTML(value=STATUS_PLACEHOLDER)
                    cp_gallery = gr.Gallery(
                        label="ColPali results", columns=ARGS.max_top_n,
                        height="auto", object_fit="contain", show_label=False,
                    )
                with gr.Column():
                    gr.Markdown("### Text chunking (OCR + nomic)")
                    tx_badge = gr.HTML(value=STATUS_PLACEHOLDER)
                    tx_gallery = gr.Gallery(
                        label="Text results", columns=ARGS.max_top_n,
                        height="auto", object_fit="contain", show_label=False,
                    )

            metrics_md = gr.Markdown(METRICS_LINE, visible=bool(QUERIES))

        with gr.Tab("Ingestion"):
            gr.Markdown(
                "### Rebuild the Qdrant index\n"
                "Full rebuild (delete + recreate). Reuses the already-loaded models. "
                "**The selected side returns no results while it rebuilds.**"
            )
            with gr.Row():
                ingest_source = gr.Radio(
                    ["ViDoRe dataset", "Upload PDF"],
                    value="ViDoRe dataset", label="Source", scale=2,
                )
                ingest_which = gr.Radio(
                    ["Both", "ColPali pages", "Text chunks"],
                    value="Both", label="Index to rebuild", scale=2,
                )
            with gr.Group() as dataset_group:
                ingest_limit = gr.Number(
                    value=ARGS.limit, precision=0,
                    label=f"Limit (pages) — from {ARGS.dataset} [{ARGS.split}]",
                )
            with gr.Group(visible=False) as pdf_group:
                ingest_pdf = gr.File(
                    label="PDF file (every page is indexed; free-text search only)",
                    file_types=[".pdf"], type="filepath",
                )
            with gr.Row():
                ingest_chunk_size = gr.Number(
                    value=ARGS.chunk_size, precision=0, label="Chunk size (words)",
                )
                ingest_chunk_overlap = gr.Number(
                    value=ARGS.chunk_overlap, precision=0, label="Chunk overlap (words)",
                )
            ingest_tess = gr.Textbox(
                value=ARGS.tesseract_cmd, label="Tesseract path (text side OCR)",
            )
            ingest_btn = gr.Button("Run ingestion", variant="primary", elem_id="run-ingestion")
            ingest_log = gr.Textbox(
                label="Progress", lines=16, max_lines=16, autoscroll=True,
                interactive=False,
            )

        with gr.Tab("Setup"):
            gr.Markdown("### Qdrant connection")
            with gr.Row():
                qdrant_status_html = gr.HTML(value=qdrant_status())
                test_btn = gr.Button("↻  Test connection", scale=0, elem_id="test-qdrant")

            gr.Markdown(
                "### Initialize a collection\n"
                "Create an empty collection."
            )
            with gr.Row():
                coll_name = gr.Textbox(value="pdf_pages", label="Collection name", scale=2)
                coll_type = gr.Radio(
                    [MULTIVECTOR_LABEL, SINGLEVECTOR_LABEL],
                    value=MULTIVECTOR_LABEL, label="Vector type", scale=2,
                )
                coll_size = gr.Number(value=128, precision=0, label="Vector size", scale=1)
            coll_recreate = gr.Checkbox(
                value=False, label="Recreate if it already exists (delete first)",
            )
            create_btn = gr.Button("Create collection", variant="primary", elem_id="create-collection")
            create_result = gr.HTML()

    # Picking a dataset query fills the box (gold becomes resolvable at retrieve).
    query_dropdown.change(on_pick_query, inputs=query_dropdown, outputs=query_box)

    search_inputs = [query_box, top_n_slider, query_dropdown]
    # Two independent events per trigger: ColPali's column updates as soon as its
    # search returns, without waiting for the (slower) text side.
    for trigger in (retrieve_btn.click, query_box.submit):
        trigger(run_colpali, inputs=search_inputs, outputs=[cp_gallery, cp_badge])
        trigger(run_text, inputs=search_inputs, outputs=[tx_gallery, tx_badge])

    # Toggle the source-specific inputs (dataset limit vs PDF upload).
    def _toggle_source(src):
        is_pdf = src == "Upload PDF"
        return gr.update(visible=not is_pdf), gr.update(visible=is_pdf)

    ingest_source.change(_toggle_source, inputs=ingest_source, outputs=[dataset_group, pdf_group])

    # Ingestion streams its log; afterwards refresh the Search dropdown to the
    # (possibly changed) query set.
    ingest_btn.click(
        run_ingestion,
        inputs=[ingest_source, ingest_which, ingest_limit, ingest_pdf,
                ingest_chunk_size, ingest_chunk_overlap, ingest_tess],
        outputs=ingest_log,
    ).then(after_ingest, outputs=[query_dropdown, metrics_md])

    # Setup tab: connection test + collection creation.
    test_btn.click(qdrant_status, outputs=qdrant_status_html)
    create_btn.click(
        init_collection,
        inputs=[coll_name, coll_type, coll_size, coll_recreate],
        outputs=[create_result, qdrant_status_html],
    )


def warmup():
    """Absorb first-call model latency with throwaway queries on both paths."""
    try:
        retrieve_colpali("warmup", 1)
        embed_query_local("warmup")
        print("[warmup] ColPali + text query paths ready.")
    except Exception as e:  # noqa: BLE001
        print(f"[warmup] skipped: {e}")


if __name__ == "__main__":
    warmup()
    print("=" * 70)
    demo.launch(
        server_name=ARGS.server_name,
        server_port=ARGS.server_port,
        share=ARGS.share,
    )
