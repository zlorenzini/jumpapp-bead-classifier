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
  "name":       "Green Silicone 10mm Sphere",
  "sku":        "GS-10-SPH",
  "image_b64":  "<base64-encoded JPEG or PNG>",
  "mime_type":  "image/jpeg",
  "source":     "Temu",
  "cost":       3.49
}
```

`image_path` may be used instead of `image_b64` to reference a file already
on disk (e.g. a frame saved by `/capture`).

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

### Shopify integration

| Method | Path | Description |
|---|---|---|
| `POST` | `/shopify/tag` | Push detected bead tags to a Shopify product |

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
    ├── state.py          # Shared mutable state (models_dir)
    ├── shopify_client.py # Shopify Admin API integration
    ├── requirements.txt
    └── routes/
        ├── add_bead.py   # POST /add-bead
        ├── train.py      # POST|GET /train, POST /train/{id}/stop
        └── bundles.py    # GET /bundles
```

---

## Interactive API docs

FastAPI generates interactive documentation automatically:

- **Swagger UI** — `http://localhost:8000/docs`
- **ReDoc** — `http://localhost:8000/redoc`
