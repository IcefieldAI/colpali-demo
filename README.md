# ColPali vs Text-Chunking Retrieval - Demo

A single-screen [Gradio](https://www.gradio.app/) app that compares two document
retrieval pipelines over the same pages, side by side:

- **ColPali** - visual, multivector (MaxSim) retrieval. The page *image* is embedded
  directly with a vision-language model; no OCR.
- **Text chunking** - the traditional RAG baseline: OCR the page → chunk the text →
  embed chunks with `nomic-embed-text` → retrieve → collapse chunks back to pages.

Enter a query and the app renders the top-N retrieved **page images** from each
pipeline with their scores, and HIT/MISS badges when the gold page is known. The point
is to make the quality gap visible: ColPali returns the correct page - especially for
tables and figures - where text chunking often misses.

Query embedding for both pipelines runs in-process (ColPali + nomic via
`transformers`), so no embedding server (e.g. Ollama) is needed at runtime - only
Qdrant.

## Interface

Three tabs:

- **Search** - a query box, and (for a dataset source) a dropdown of the dataset's own
  gold queries. Shows both pipelines' top-N page images with per-side retrieval latency
  and, when the gold page is known, a HIT/MISS badge plus a highlighted gold page.
- **Ingestion** - (re)build the Qdrant index from either a **ViDoRe dataset subset**
  (gold queries available → HIT/MISS) or an **uploaded PDF** (free-text search only).
  Reuses the already-loaded models and streams a progress log. Full rebuild
  (delete + recreate).
- **Setup** - test the Qdrant connection and initialize a collection.

The last-used source is persisted under `app_state/` (including rendered PDF page
images), so the app restores it on restart and hides the dataset-specific UI when the
source is a PDF.

## Requirements

This stack is version-sensitive - pin it as shown below.

- **Python 3.12**, managed with [uv](https://docs.astral.sh/uv/)
- A running **Qdrant** at `http://localhost:6333`
- **Tesseract OCR** - only used by the text-side *ingestion* (page OCR)
- An NVIDIA GPU is recommended (CUDA build of PyTorch); CPU works but is slow

> **Gotcha:** installing `gradio` upgrades `huggingface-hub` to 1.x, which breaks the
> pinned `transformers` 4.46 (`transformers`/`colpali-engine` then fail to import). The
> last install step pins `huggingface-hub` back below 1.0.

## Setup

```powershell
# 1. Tooling (Windows; see the uv / Docker docs for other platforms)
winget install --id=astral-sh.uv -e
winget install -e --id UB-Mannheim.TesseractOCR          # text-side OCR ingestion
# Docker Desktop: https://docs.docker.com/desktop/setup/install/windows-install/

# 2. Virtual environment
uv venv --python 3.12
.venv\Scripts\activate                                   # Windows
source .venv/bin/activate                              # Linux/macOS

# 3. Dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "colpali-engine>=0.3.1,<0.3.5" "transformers>=4.46.1,<4.47.0"
uv pip install qdrant-client datasets pillow accelerate gradio pytesseract pymupdf einops

# 4. Pin huggingface-hub back below 1.0 (gradio bumps it and breaks transformers 4.46)
uv pip install "huggingface-hub==0.36.2"
```

Sanity check:

```powershell
python -c "import torch; from importlib.metadata import version; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', version('transformers'), 'hf-hub', version('huggingface-hub'))"
```

Start Qdrant:

```powershell
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Running

```powershell
python app.py            # serves http://127.0.0.1:7860
```

Then, in the browser:

1. **Setup** → *Test connection* to confirm Qdrant is reachable.
2. **Ingestion** → choose a source (ViDoRe dataset *or* a PDF), pick what to build
   (ColPali pages / text chunks / both), and *Run ingestion*.
3. **Search** → type a query (or pick a dataset query) and compare the pipelines.

Useful flags (`python app.py --help` for the full list):

| Flag | Purpose |
|------|---------|
| `--qdrant-url` | Qdrant endpoint (default `http://localhost:6333`) |
| `--pages-collection` / `--text-collection` | collection names (`pdf_pages` / `pdf_text`) |
| `--dataset` / `--split` / `--limit` | ViDoRe source selection |
| `--text-model` | query/chunk embedder (default `nomic-ai/nomic-embed-text-v1.5`) |
| `--tesseract-cmd` | path to `tesseract.exe` for OCR ingestion |

## Metrics

The aggregate panel shows precomputed retrieval quality on a DocVQA subset:

- **hit-rate@k** - fraction of queries whose gold page is in the top-k.
- **MRR@k** - Mean Reciprocal Rank: the mean of `1/rank` of the gold page (rank 1 → 1.0,
  rank 2 → 0.5, …), rewarding the correct page appearing *higher* in the list.

## License

MIT - see `LICENSE`.
