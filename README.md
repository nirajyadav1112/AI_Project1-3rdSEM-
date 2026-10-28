# AI-Based Smart Waste Management System

A local prototype for bin readings, image-based waste classification, full-bin alerts, collection forecasting, and a simple dashboard.

## 1. Start the dashboard and API

Use Python 3.10 or newer. From PowerShell in this project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API documentation is at <http://127.0.0.1:8000/docs>. Data is stored in `waste.db`; set `WASTE_DB_PATH` to use another path.

The sensor-reading form and dashboard work without an AI model. Image uploads are validated and recorded, but are marked **unclassified / needs review** until a trained checkpoint is available. The uploaded image itself is not retained.

## 2. Train the image classifier

The starter CNN classifies the overall image into five material labels. Organize a labeled dataset into separate training and validation folders:

```text
data/
  train/
    plastic/
    paper/
    metal/
    organic/
    other/
  val/
    plastic/
    paper/
    metal/
    organic/
    other/
```

Install the machine-learning dependency and train:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe train_model.py --data data --epochs 10
```

The trained checkpoint is saved to `models/waste_cnn.pt`, which the app loads on startup. Restart the app after training. Do not treat this small CNN as production-ready; use a larger, representative dataset, compare against a pretrained model, and review per-class metrics before deployment.

## 3. API examples

Record a sensor reading (creates an alert at 90% fill):

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/readings `
  -ContentType "application/json" `
  -Body '{"bin_id":"BIN-001","fill_pct":93}'
```

Upload an image, optionally with its fill level:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/records/upload `
  -F "bin_id=BIN-001" -F "fill_pct=93" -F "image=@C:\path\to\waste.jpg"
```

Useful endpoints:

- `GET /api/health` — API and classifier readiness.
- `GET /api/dashboard` — summary counts and recent records.
- `GET /api/records` — recent waste records.
- `GET /api/alerts` — open full-bin alerts.
- `PATCH /api/alerts/{id}` — resolve an alert with `{"status":"resolved"}`.
- `GET /api/forecast` — simple fill-rate extrapolation when two readings exist.
- `GET /api/route` — nearest-neighbor suggestion for full bins with coordinates.

To configure bin coordinates for the route demo, set `BIN_LOCATIONS_JSON` before starting the app:

```powershell
$env:BIN_LOCATIONS_JSON = '{"BIN-001":{"lat":12.97,"lon":77.59},"BIN-002":{"lat":12.98,"lon":77.60}}'
```

The route endpoint uses straight-line distance, not live road-network travel times, vehicle capacity, or traffic. It is a demo heuristic, not a production optimizer. Sensor data can be sent to `POST /api/readings`; an ESP32 or gateway can call that endpoint over the local network.

## 4. Data and model notes

Possible starting datasets include TrashNet for common material classes and TACO for litter detection research. Verify license and local applicability before use. Dataset labels may not match municipal recycling policies; adjust `MATERIAL_CATEGORIES` in `app.py` to local rules. A single-image classifier does not locate multiple items and cannot reliably assess contamination; use an object detector trained on bounding-box annotations for that requirement.

## 5. Evaluation plan

- Classification: per-class precision, recall, F1, macro-F1, confusion matrix, and human-review rate.
- Fill sensing: mean absolute error against manual fill measurements.
- Forecasting: time-to-full MAE and missed-overflow rate.
- Collection routes: distance, time, fuel estimate, pickups per route, and missed bins versus the current route.
- Operations: alert latency, false-alert rate, worker corrections, and collection adherence.
- Pilot impact: overflow incidents, unnecessary trips, contamination, and cost per pickup.

Split evaluation data by bin/location or collection period to reduce leakage. Establish a baseline with municipal staff before setting targets or claiming impact.

## 6. Prototype cost estimate

Indicative costs vary by supplier and location: ESP32 plus ultrasonic sensor roughly ₹500–₹1,500 per bin, enclosure/power/wiring ₹500–₹2,000, and an optional camera ₹1,000–₹4,000. A 3–5-bin demo may cost roughly ₹5,000–₹25,000 excluding staff time, connectivity, hosting, installation, and vehicle operations. Start with phone uploads before installing cameras.

## 7. Production hardening

Before deployment, add user authentication and roles, HTTPS, rate limits, sensor identity/authentication, retention controls, database backups, notification delivery, monitoring, road-network routing, and human review workflows. Do not expose this prototype directly to the public internet.
