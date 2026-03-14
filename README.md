# Bead Classifier JumpApp

A portable bead-identification system that runs on a JumpNet hub.  It uses a
**ResNet-50 backbone** to embed reference photos of beads into a MongoDB data
lake, identifies beads in new product images via similarity search, and can
optionally train a **MobileNetV2 classifier** from the collected data for
higher-accuracy inference.

---

## How it works

```
Camera ──► JumpNet /capture ──► POST /capture ──► ResNet-50 embedder
                                                       │
                           MongoDB bead data lake ◄────┘
                                   │
                          POST /infer (query image)
                                   │
                    ┌──────────────┴──────────────┐
                    │ Trained MobileNetV2 model    │  (if /train has been run)
                    │ (models/current/model.pth)   │
                    └──────────────┬──────────────┘
                                   │  fallback for beads added after last train
                    ┌──────────────┴──────────────┐
                    │ ResNet-50 cosine similarity  │
                    └─────────────────────────────┘
                                   │
                            ranked matches
```

1. **Ingest** – `POST /add-bead` stores a reference photo + metadata for each bead SKU.
2. **Identify** – `POST /infer` or `POST /capture` returns the best-matching bead SKUs with confidence scores.
3. **Train** – `POST /train` fine-tunes a MobileNetV2 classifier on all collected images and saves it to `models/current/`.  Subsequent inference uses the trained model automatically.

---

## Quick start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | venv recommended |
| MongoDB | `mongod` running on `localhost:27017` |
| JumpNet hub | running on `http://localhost:4080` (for `/capture` and `/preview`) |
| ffmpeg | required only for camera capture |

### Install

```bash
cd jumpapp-bead-classifier/server
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — the only required change for local use is usually nothing;
# set SHOPIFY_* only if you want to push tags to Shopify.
```

### Run

```bash
cd server
python main.py
# → http://127.0.0.1:8000
# → http://127.0.0.1:8000/docs  (interactive API docs)
```

The server is also managed as a **systemd service** (`bead-classifier.service`)
on the JumpNet hub — it starts automatically on boot alongside `jumpnet.service`
and `mongod.service`.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DB` | `bead_classifier` | Database name |
| `MONGODB_COLLECTION` | `beads` | Collection name |
| `CONFIDENCE_THRESHOLD` | `0.65` | Minimum score to include a match |
| `TOP_K` | `5` | Maximum matches returned per query |
| `JUMPNET_URL` | `http://localhost:4080` | URL of the JumpNet gateway |
| `CAPTURE_DEVICE` | `/dev/video0` | v4l2 camera device |
| `CAPTURE_RESOLUTION` | `960x720` | Capture resolution |
| `CAPTURE_INPUT_FORMAT` | `yuyv422` | `mjpeg` or `yuyv422` |
| `CAPTURE_WARMUP` | `10` | Frames discarded for AE/AWB settle |
| `CAPTURE_FPS` | *(auto)* | Override camera framerate |
| `SHOPIFY_STORE` | — | `my-store.myshopify.com` |
| `SHOPIFY_API_TOKEN` | — | Private app token (`shpat_…`) |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8000` | Bind port |

Set `JUMPAPP_ROOT` to override the storage root (models, dataset, endpoints
directories) — useful when the USB drive is mounted at a non-default path.

---

## API reference

### Health & discovery

| Method | Path | Description |
|---|---|---|
| `GET` | `/status` | Health check, bead count, DB/Shopify readiness |
| `GET` | `/bundles` | Backbone descriptor + embedding-store statistics |

### Bead data lake

| Method | Path | Description |
|---|---|---|
| `GET` | `/beads` | List all beads (no image data) |
| `GET` | `/beads/{bead_id}` | Fetch one bead including images |
| `DELETE` | `/beads/{bead_id}` | Remove a bead |
| `POST` | `/add-bead` | Ingest a bead angle — JSON body |
| `POST` | `/add-bead/upload` | Ingest a bead angle — multipart form |

#### `POST /add-bead` — JSON body

```json
{
  "name":              "Green Silicone 10mm Hexagon",
  "sku":               "GS-10-HEX",
  "image_b64":         "<base64-encoded JPEG or PNG>",
  "mime_type":         "image/jpeg",
  "source":            "Temu",
  "cost":              3.49,
  "material_category": "silicone",
  "material":          "silicone",
  "color":             "dusty rose",
  "color_family":      "pink",
  "shape":             "hexagon",
  "size_mm":           10.0,
  "finish":            "matte",
  "hole_type":         "center-drilled",
  "focal_bead":        false,
  "tags":              ["silicone", "hexagon"]
}
```

`image_path` may be used instead of `image_b64` to reference a file already
on disk (e.g. a frame saved by `/capture`).  All ontology fields are optional.

**Ontology field values**

| Field | Allowed values |
|---|---|
| `material_category` | `glass` \| `silicone` \| `acrylic` \| `metal` \| `ceramic` \| `wood` \| `gemstone` \| `organic` \| `resin` \| `rhinestone` \| `fabric` \| `other` |
| `material` | free text — e.g. `"czech glass"`, `"sterling silver"`, `"rose quartz"`, `"polymer clay"` |
| `color` | free text — e.g. `"dusty rose"`, `"cobalt blue"` |
| `color_family` | `red` \| `orange` \| `yellow` \| `green` \| `blue` \| `purple` \| `pink` \| `brown` \| `white` \| `black` \| `gray` \| `gold` \| `silver` \| `multicolor` |
| `shape` | `round` \| `oval` \| `tube` \| `flat` \| `bicone` \| `hexagon` \| `rondelle` \| `faceted` \| `drop` \| `cube` \| `heart` \| `star` \| `letter` \| `barrel` \| `nugget` \| `coin` \| `other` |
| `size_mm` | positive number — nominal diameter in mm |
| `finish` | `glossy` \| `matte` \| `transparent` \| `frosted` \| `metallic` \| `iridescent` \| `pearlized` \| `crackle` \| `painted` \| `etched` \| `glow` \| `uv-reactive` \| `other` |
| `hole_type` | `center-drilled` \| `large-hole` \| `top-drilled` \| `side-drilled` |

#### `POST /add-bead/upload` — multipart fields

| Field | Type | Required |
|---|---|---|
| `name` | text | ✓ |
| `sku` | text | ✓ |
| `image` | file | ✓ (or `image_path`) |
| `image_path` | text | ✓ (or `image`) |
| `source` | text | — default `"Unknown"` |
| `cost` | float | — default `0.0` |
| `mime_type` | text | — default `"image/jpeg"` |
| `material_category` | text | — |
| `material` | text | — |
| `color` | text | — |
| `color_family` | text | — |
| `shape` | text | — |
| `size_mm` | float | — |
| `finish` | text | — |
| `hole_type` | text | — |
| `focal_bead` | bool | — |
| `is_3d` | bool | — |
| `focal_description` | text | — |

### Inference

| Method | Path | Description |
|---|---|---|
| `POST` | `/infer` | Identify beads in an uploaded image |
| `POST` | `/capture` | Capture a frame via JumpNet, then infer |
| `GET` | `/preview` | Capture a frame via JumpNet, return image only |

#### `POST /infer`

Multipart `form-data` with a single `image` file field (JPEG / PNG).

**Response**

```json
{
  "matches": [
    { "bead_id": "GS-10-SPH", "name": "Green Silicone 10mm Sphere", "confidence": 0.9123 }
  ],
  "image_filename": "photo.jpg"
}
```

#### `POST /capture`

Optional JSON body:

```json
{
  "device":       "/dev/video0",
  "resolution":   "960x720",
  "fps":          15,
  "warmup":       10,
  "input_format": "yuyv422"
}
```

Returns matched beads plus the raw captured `image_b64`.

### Training

| Method | Path | Description |
|---|---|---|
| `POST` | `/train` | Start a training job (async) |
| `GET` | `/train` | List all training jobs |
| `GET` | `/train/{id}` | Job status, per-epoch logs, final result |
| `POST` | `/train/{id}/stop` | Request graceful stop |

#### `POST /train` — request body (all fields optional)

```json
{
  "bundle_id":  "current",
  "epochs":     10,
  "img_size":   224,
  "batch_size": 16,
  "lr":         0.0001
}
```

**Response** — returns immediately; poll `GET /train/{id}` for progress.

```json
{
  "id":         "a3f1…",
  "status":     "queued",
  "bundle_id":  "current",
  "epochs":     10,
  "created_at": "2026-03-14T12:00:00Z"
}
```

**Job statuses:** `queued` → `running` → `done` | `error` | `stopped`

#### `GET /train/{id}` — full job response

```json
{
  "id":       "a3f1…",
  "status":   "done",
  "progress": { "epoch": 10, "epochs": 10, "train_loss": 0.12, "val_accuracy": 0.94 },
  "result": {
    "bundle_id": "current",
    "classes":   ["GS-10-SPH", "FB-BLUE-001"],
    "accuracy":  0.94,
    "epochs":    10,
    "images":    42
  },
  "logs": ["Using device: cuda", "Exported 42 images…", "Epoch 1/10 loss=0.45 val_acc=0.71", "…"],
  "error": null
}
```

> **Training requirements:** at least **2 distinct bead SKUs** with at least
> one image each must be in the database before training can start.

### Acquisition

| Method | Path | Description |
|---|---|---|
| `POST` | `/acquire` | Search the internet for bead images and ingest them |

Searches for bead images matching a query string, extracts metadata from text
context (title, alt text, caption), and ingests each image into the data lake
via the same `/add-bead` pipeline.

**What it does not do:** it does not crawl websites, parse HTML, or follow
links.  It fetches only the direct image URLs returned by the search API.

#### `POST /acquire` — request body

```json
{
  "query":       "10mm silicone hexagon dusty rose",
  "max_results": 5
}
```

**Response**

```json
{
  "query": "10mm silicone hexagon dusty rose",
  "ingested": [
    {
      "sku":          "10mm-silicone-hexagon-dusty-rose",
      "name":         "10mm Silicone Hexagon Dusty Rose Bead",
      "image_url":    "https://example.com/bead.jpg",
      "embedding_id": "…"
    }
  ],
  "skipped": [
    { "image_url": "https://example.com/bead2.jpg", "reason": "HTTP 403" }
  ],
  "errors": []
}
```

#### Pluggable backends

Both the search API and the LLM metadata extractor are abstract interfaces
defined in `server/acquisition.py`.  The defaults are stubs (return nothing)
so the endpoint is always safe to call but does nothing without a real backend.

**Wiring in a real search client:**

```python
# In acquisition.py or a deployment config module:
from acquisition import SearchResult

class GoogleImageSearchClient(SearchClient):
    def __init__(self, api_key: str, cx: str):
        self._api_key = api_key
        self._cx = cx

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        # Call the Google Custom Search JSON API
        ...
```

Pass it to `acquire_beads()`:

```python
from acquisition import acquire_beads, GoogleImageSearchClient

summary = await acquire_beads(
    query="10mm silicone hexagon dusty rose",
    max_results=5,
    search_client=GoogleImageSearchClient(api_key="…", cx="…"),
)
```

**Wiring in a real LLM extractor:**

```python
class OpenAIExtractor(LLMExtractor):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._api_key = api_key
        self._model = model

    async def extract(self, *, page_title, alt_text, caption, surrounding_text, query):
        # Call the OpenAI chat completions API
        ...
```

### Shopify integration

```json
{
  "product_id": "123456789",
  "bead_ids":   ["GS-10-SPH", "FB-BLUE-001"],
  "replace":    false
}
```

Tags are written as `bead:<sku>` so they can be filtered in Shopify.
Requires `SHOPIFY_STORE` and `SHOPIFY_API_TOKEN` to be set in `.env`.

---

## Project layout

```
jumpapp-bead-classifier/
├── jumpapp.json          # JumpApp identity & capability declaration
├── .env.example          # Environment variable template
├── dataset/
│   ├── labels.json       # Seed bead definitions
│   └── schema.json       # Bead document schema
├── endpoints/            # Hot-loadable endpoint modules (USB drive)
├── models/
│   └── current/          # model.pth + metadata.json written by /train
└── server/
    ├── main.py           # FastAPI app + lifespan + all route wiring
    ├── config.py         # Settings loaded from env / .env
    ├── db.py             # Motor/MongoDB data-lake client
    ├── embedder.py       # ResNet-50 embedding backbone
    ├── inference.py      # Two-tier inference (classifier + cosine fallback)
    ├── trainer.py        # Background MobileNetV2 training job runner
    ├── acquisition.py    # Acquisition pipeline: SearchClient + LLMExtractor interfaces
    ├── state.py          # Shared mutable state (models_dir)
    ├── shopify_client.py # Shopify Admin API integration
    ├── requirements.txt
    └── routes/
        ├── add_bead.py   # POST /add-bead
        ├── train.py      # POST|GET /train, POST /train/{id}/stop
        ├── bundles.py    # GET /bundles
        └── acquire.py    # POST /acquire
```

---

## Interactive API docs

FastAPI generates interactive documentation automatically:

- **Swagger UI** — `http://localhost:8000/docs`
- **ReDoc** — `http://localhost:8000/redoc`
