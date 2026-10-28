from __future__ import annotations

import io
import json
import math
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Iterator

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from classifier import WasteClassifier

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("WASTE_DB_PATH", BASE_DIR / "waste.db"))
MODEL_PATH = Path(os.environ.get("WASTE_MODEL_PATH", BASE_DIR / "models" / "waste_cnn.pt"))
ALERT_THRESHOLD = 90.0
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MATERIAL_CATEGORIES = {
    "plastic": "recyclable",
    "paper": "recyclable",
    "metal": "recyclable",
    "organic": "biodegradable",
    "other": "non-recyclable",
}


@contextmanager
def connect_db() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()


def initialize_db() -> None:
    with connect_db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS waste_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bin_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                source TEXT NOT NULL,
                material TEXT,
                category TEXT,
                confidence REAL,
                fill_pct REAL,
                needs_review INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_records_bin_time
                ON waste_records(bin_id, recorded_at);
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bin_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open'
            );
            """
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_db()
    yield


app = FastAPI(title="Smart Waste Management", lifespan=lifespan)
classifier = WasteClassifier(MODEL_PATH)


class ReadingInput(BaseModel):
    bin_id: str = Field(min_length=1, max_length=100)
    fill_pct: float = Field(ge=0, le=100)


class AlertUpdate(BaseModel):
    status: str


def insert_record(
    *,
    bin_id: str,
    source: str,
    fill_pct: float | None,
    material: str | None = None,
    confidence: float | None = None,
    needs_review: bool = False,
) -> tuple[int, str, bool]:
    recorded_at = datetime.now(timezone.utc).isoformat()
    category = MATERIAL_CATEGORIES.get(material) if material else None
    alert_created = False

    with connect_db() as connection:
        cursor = connection.execute(
            """
            INSERT INTO waste_records
                (bin_id, recorded_at, source, material, category, confidence, fill_pct, needs_review)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bin_id,
                recorded_at,
                source,
                material,
                category,
                confidence,
                fill_pct,
                int(needs_review),
            ),
        )
        if fill_pct is not None and fill_pct >= ALERT_THRESHOLD:
            existing_alert = connection.execute(
                "SELECT 1 FROM alerts WHERE bin_id = ? AND status = 'open' LIMIT 1",
                (bin_id,),
            ).fetchone()
            if existing_alert is None:
                connection.execute(
                    "INSERT INTO alerts (bin_id, recorded_at, reason) VALUES (?, ?, ?)",
                    (bin_id, recorded_at, f"Bin is {fill_pct:.0f}% full"),
                )
                alert_created = True
        return int(cursor.lastrowid), recorded_at, alert_created


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "classifier_available": classifier.available,
        "classifier_message": classifier.message,
    }


@app.post("/api/readings")
def create_reading(reading: ReadingInput) -> dict[str, object]:
    bin_id = reading.bin_id.strip()
    if not bin_id:
        raise HTTPException(status_code=422, detail="bin_id must not be blank")

    record_id, recorded_at, alert_created = insert_record(
        bin_id=bin_id,
        source="sensor",
        fill_pct=reading.fill_pct,
    )
    return {
        "id": record_id,
        "bin_id": bin_id,
        "recorded_at": recorded_at,
        "fill_pct": reading.fill_pct,
        "alert_created": alert_created,
    }


@app.post("/api/records/upload")
async def upload_record(
    bin_id: Annotated[str, Form(min_length=1, max_length=100)],
    image: Annotated[UploadFile, File()],
    fill_pct: Annotated[float | None, Form(ge=0, le=100)] = None,
) -> dict[str, object]:
    bin_id = bin_id.strip()
    if not bin_id:
        raise HTTPException(status_code=422, detail="bin_id must not be blank")

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        material, confidence = classifier.predict(io.BytesIO(image_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    needs_review = (
        material not in MATERIAL_CATEGORIES
        or confidence is None
        or confidence < 0.70
    )
    record_id, recorded_at, alert_created = insert_record(
        bin_id=bin_id,
        source="image",
        fill_pct=fill_pct,
        material=material,
        confidence=confidence,
        needs_review=needs_review,
    )

    return {
        "id": record_id,
        "bin_id": bin_id,
        "recorded_at": recorded_at,
        "material": material,
        "category": MATERIAL_CATEGORIES.get(material) if material else None,
        "confidence": confidence,
        "needs_review": needs_review,
        "classifier_available": classifier.available,
        "classifier_message": classifier.message,
        "fill_pct": fill_pct,
        "alert_created": alert_created,
    }


@app.get("/api/records")
def list_records(limit: int = 50) -> list[dict[str, object]]:
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    with connect_db() as connection:
        rows = connection.execute(
            "SELECT * FROM waste_records ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/alerts")
def list_alerts() -> list[dict[str, object]]:
    with connect_db() as connection:
        rows = connection.execute(
            "SELECT * FROM alerts WHERE status = 'open' ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.patch("/api/alerts/{alert_id}")
def update_alert(alert_id: int, update: AlertUpdate) -> dict[str, object]:
    if update.status not in {"open", "resolved"}:
        raise HTTPException(status_code=422, detail="status must be open or resolved")
    with connect_db() as connection:
        cursor = connection.execute(
            "UPDATE alerts SET status = ? WHERE id = ?", (update.status, alert_id)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Alert not found")
    return {"id": alert_id, "status": update.status}


@app.get("/api/dashboard")
def dashboard_data() -> dict[str, object]:
    with connect_db() as connection:
        summary = connection.execute(
            """
            SELECT COUNT(*) AS record_count,
                   COUNT(DISTINCT bin_id) AS bin_count,
                   SUM(CASE WHEN needs_review = 1 THEN 1 ELSE 0 END) AS review_count
            FROM waste_records
            """
        ).fetchone()
        material_rows = connection.execute(
            """
            SELECT COALESCE(category, 'unclassified') AS category, COUNT(*) AS count
            FROM waste_records
            GROUP BY COALESCE(category, 'unclassified')
            """
        ).fetchall()
        alert_count = connection.execute(
            "SELECT COUNT(*) AS count FROM alerts WHERE status = 'open'"
        ).fetchone()["count"]
        recent_rows = connection.execute(
            "SELECT * FROM waste_records ORDER BY id DESC LIMIT 10"
        ).fetchall()

    return {
        "record_count": summary["record_count"] or 0,
        "bin_count": summary["bin_count"] or 0,
        "review_count": summary["review_count"] or 0,
        "open_alert_count": alert_count,
        "categories": {row["category"]: row["count"] for row in material_rows},
        "recent_records": [dict(row) for row in recent_rows],
        "classifier_available": classifier.available,
        "classifier_message": classifier.message,
    }


@app.get("/api/forecast")
def forecast() -> list[dict[str, object]]:
    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT bin_id, recorded_at, fill_pct
            FROM waste_records
            WHERE fill_pct IS NOT NULL
            ORDER BY bin_id, recorded_at DESC
            """
        ).fetchall()

    readings_by_bin: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        readings_by_bin.setdefault(row["bin_id"], []).append(row)

    predictions: list[dict[str, object]] = []
    for bin_id, readings in readings_by_bin.items():
        latest = readings[0]
        minutes_to_full: float | None = None
        if len(readings) >= 2:
            previous = readings[1]
            latest_time = datetime.fromisoformat(latest["recorded_at"])
            previous_time = datetime.fromisoformat(previous["recorded_at"])
            elapsed_minutes = (latest_time - previous_time).total_seconds() / 60
            fill_increase = latest["fill_pct"] - previous["fill_pct"]
            if elapsed_minutes > 0 and fill_increase > 0 and latest["fill_pct"] < 100:
                minutes_to_full = (
                    (100 - latest["fill_pct"]) * elapsed_minutes / fill_increase
                )
        predictions.append(
            {
                "bin_id": bin_id,
                "fill_pct": latest["fill_pct"],
                "estimated_minutes_to_full": (
                    round(minutes_to_full, 1) if minutes_to_full is not None else None
                ),
                "forecast_status": (
                    "estimated" if minutes_to_full is not None else "insufficient_trend_data"
                ),
            }
        )
    return predictions


@app.get("/api/route")
def suggest_route(
    start_lat: float = Query(-27.4698, ge=-90, le=90),
    start_lon: float = Query(153.0251, ge=-180, le=180),
    max_stops: int = Query(10, ge=1, le=100),
) -> dict[str, object]:
    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT bin_id, fill_pct
            FROM waste_records
            WHERE fill_pct IS NOT NULL
            ORDER BY bin_id, recorded_at DESC
            """
        ).fetchall()
    latest_by_bin: dict[str, float] = {}
    for row in rows:
        latest_by_bin.setdefault(row["bin_id"], row["fill_pct"])

    # Coordinates are optional bin metadata supplied through environment JSON.
    raw_locations = os.environ.get("BIN_LOCATIONS_JSON", "{}")
    try:
        locations = json.loads(raw_locations)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="BIN_LOCATIONS_JSON is invalid JSON") from exc
    if not isinstance(locations, dict):
        raise HTTPException(status_code=500, detail="BIN_LOCATIONS_JSON must be an object")
    for bin_id, coordinates in locations.items():
        if not isinstance(bin_id, str) or not isinstance(coordinates, dict):
            raise HTTPException(
                status_code=500,
                detail="Each BIN_LOCATIONS_JSON entry must map a bin ID to coordinates",
            )
        latitude, longitude = coordinates.get("lat"), coordinates.get("lon")
        if (
            isinstance(latitude, bool)
            or isinstance(longitude, bool)
            or not isinstance(latitude, (int, float))
            or not isinstance(longitude, (int, float))
            or not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
        ):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid latitude or longitude configured for bin {bin_id}",
            )

    pending = [
        (bin_id, fill_pct, locations.get(bin_id))
        for bin_id, fill_pct in latest_by_bin.items()
        if fill_pct >= ALERT_THRESHOLD and bin_id in locations
    ]
    route: list[dict[str, object]] = []
    current_lat, current_lon = start_lat, start_lon
    while pending and len(route) < max_stops:
        next_bin = min(
            pending,
            key=lambda item: _haversine_km(
                current_lat, current_lon, float(item[2]["lat"]), float(item[2]["lon"])
            ),
        )
        pending.remove(next_bin)
        bin_id, fill_pct, coordinates = next_bin
        route.append(
            {
                "bin_id": bin_id,
                "fill_pct": fill_pct,
                "lat": coordinates["lat"],
                "lon": coordinates["lon"],
            }
        )
        current_lat, current_lon = float(coordinates["lat"]), float(coordinates["lon"])

    return {
        "method": "nearest-neighbor distance heuristic",
        "route": route,
        "unrouted_full_bins": len(pending),
        "message": (
            "Add bin coordinates to BIN_LOCATIONS_JSON to include them in route suggestions."
            if not locations
            else "This is a prototype route suggestion, not a road-network optimized route."
        ),
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    return earth_radius_km * 2 * math.asin(math.sqrt(haversine))


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Waste Management</title>
  <style>
    body { font: 16px system-ui, sans-serif; max-width: 1050px; margin: 2rem auto; padding: 0 1rem; color: #18352b; background: #f3f8f4; }
    h1 { margin-bottom: .25rem; } .muted { color: #587065; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap: 1rem; margin: 1.5rem 0; }
    .card, form, table { background: white; border: 1px solid #dce8df; border-radius: 12px; padding: 1rem; }
    .value { display: block; font-size: 1.8rem; font-weight: 700; }
    .forms { display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap: 1rem; }
    label { display: block; margin: .6rem 0 .25rem; } input, button { font: inherit; padding: .55rem; }
    input { width: calc(100% - 1.2rem); } button { margin-top: .8rem; color: white; background: #18764d; border: 0; border-radius: 6px; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; padding: 0; overflow: hidden; } th, td { padding: .65rem; border-bottom: 1px solid #e5eee7; text-align: left; }
    .notice { padding: .75rem; background: #fff5d6; border-radius: 8px; } pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Smart Waste Management</h1>
  <p class="muted">Bin monitoring, waste classification, alerts, and collection planning.</p>
  <p id="model-status" class="notice">Loading model status…</p>
  <section class="grid">
    <div class="card">Records <span id="records" class="value">—</span></div>
    <div class="card">Bins seen <span id="bins" class="value">—</span></div>
    <div class="card">Open alerts <span id="alerts" class="value">—</span></div>
    <div class="card">Needs review <span id="review" class="value">—</span></div>
  </section>
  <section class="forms">
    <form id="reading-form">
      <h2>Record bin reading</h2>
      <label for="reading-bin">Bin ID</label><input id="reading-bin" required>
      <label for="reading-fill">Fill level (%)</label><input id="reading-fill" type="number" min="0" max="100" required>
      <button>Save sensor reading</button>
    </form>
    <form id="image-form">
      <h2>Classify waste photo</h2>
      <label for="image-bin">Bin ID</label><input id="image-bin" required>
      <label for="image-fill">Optional fill level (%)</label><input id="image-fill" type="number" min="0" max="100">
      <label for="image-file">Image (max 8 MB)</label><input id="image-file" type="file" accept="image/*" required>
      <button>Upload image</button>
    </form>
  </section>
  <p id="message" role="status"></p>
  <h2>Collection forecast</h2><pre id="forecast">Loading…</pre>
  <h2>Recent records</h2>
  <table><thead><tr><th>Bin</th><th>Source</th><th>Material</th><th>Fill</th><th>Review</th><th>Recorded</th></tr></thead><tbody id="records-table"></tbody></table>
  <h2>Open alerts</h2><ul id="alerts-list"></ul>
<script>
const message = document.querySelector("#message");
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}
async function refresh() {
  const data = await request("/api/dashboard");
  document.querySelector("#records").textContent = data.record_count;
  document.querySelector("#bins").textContent = data.bin_count;
  document.querySelector("#alerts").textContent = data.open_alert_count;
  document.querySelector("#review").textContent = data.review_count;
  document.querySelector("#model-status").textContent = data.classifier_message;
  document.querySelector("#records-table").innerHTML = data.recent_records.map(row =>
    `<tr><td>${escapeHtml(row.bin_id)}</td><td>${escapeHtml(row.source)}</td><td>${escapeHtml(row.material || "unclassified")}</td><td>${row.fill_pct == null ? "—" : row.fill_pct + "%"}</td><td>${row.needs_review ? "Yes" : "No"}</td><td>${escapeHtml(row.recorded_at)}</td></tr>`
  ).join("");
  const alerts = await request("/api/alerts");
  document.querySelector("#alerts-list").innerHTML = alerts.map(alert =>
    `<li>${escapeHtml(alert.bin_id)} — ${escapeHtml(alert.reason)} <button onclick="resolveAlert(${alert.id})">Resolve</button></li>`
  ).join("") || "<li>No open alerts</li>";
  const forecasts = await request("/api/forecast");
  document.querySelector("#forecast").textContent = JSON.stringify(forecasts, null, 2);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
}
async function resolveAlert(id) {
  try { await request(`/api/alerts/${id}`, {method:"PATCH", headers:{"Content-Type":"application/json"}, body:JSON.stringify({status:"resolved"})}); await refresh(); }
  catch (error) { message.textContent = error.message; }
}
document.querySelector("#reading-form").addEventListener("submit", async event => {
  event.preventDefault();
  try {
    const data = await request("/api/readings", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({bin_id:document.querySelector("#reading-bin").value, fill_pct:Number(document.querySelector("#reading-fill").value)})});
    message.textContent = data.alert_created ? "Reading saved. A full-bin alert was created." : "Reading saved.";
    event.target.reset(); await refresh();
  } catch (error) { message.textContent = error.message; }
});
document.querySelector("#image-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = new FormData();
  form.append("bin_id", document.querySelector("#image-bin").value);
  const fill = document.querySelector("#image-fill").value;
  if (fill !== "") form.append("fill_pct", fill);
  form.append("image", document.querySelector("#image-file").files[0]);
  try {
    const data = await request("/api/records/upload", {method:"POST", body:form});
    message.textContent = data.classifier_available ? `Detected ${data.material} (${Math.round(data.confidence * 100)}% confidence).` : data.classifier_message;
    event.target.reset(); await refresh();
  } catch (error) { message.textContent = error.message; }
});
refresh().catch(error => { message.textContent = error.message; });
</script>
</body>
</html>"""
