# Edge AI Food Identification System

## Ghanaian Food Recognition with Diabetic Dietary Recommendations

An edge-deployed AI system that photographs meals, identifies Ghanaian foods, estimates portion sizes, and provides diabetic-friendly dietary recommendations — designed to run entirely on a Raspberry Pi 5 with no cloud dependency.

---

## Features

- **Food Segmentation** — YOLOv8-seg (NCNN format on Pi, PyTorch for development) detects and segments 17 Ghanaian food classes with pixel-accurate masks
- **Depth Estimation** — MiDaS estimates food height from a single RGB image for volume calculation
- **Volume & Weight Estimation** — Fuses segmentation masks with depth maps; applies food-specific densities from the nutrition DB
- **Three-Level Recommendation Engine** (v3.3.1, source of truth: `notebooks/06_recommendation_v3.ipynb`):
  - **Level 1** — Plate composition check against the 50 % veg / 25 % protein / 25 % starch Diabetic Plate Model
  - **Level 2** — Per-starch portion check (orange-reference volume for moulded starches; GDA serving weight for spread starches)
  - **Level 3** — Per-item Glycemic Load (GL = GI × carbs_per_100g × weight_g / 10 000)
- **Flask SPA** — Single-page web app with camera capture, image upload, meal history, per-user carb tracking, and CSV/PDF export

---

## Project Structure

```
capstone/
├── notebooks/
│   └── 06_recommendation_v3.ipynb    # Three-level recommendation engine (source of truth)
│
├── pipeline/                         # Development pipeline (mirrors pi_deploy/pipeline/)
│   ├── nutrition.py                  # Recommendation engine + nutrition lookup
│   ├── volume.py                     # Volume, weight & GL estimation
│   ├── depth.py                      # MiDaS depth estimator
│   ├── segmentation.py               # YOLOv8 model wrapper
│   └── camera.py                     # Camera interface
│
├── pi_deploy/                        # Raspberry Pi deployment package
│   ├── app.py                        # Flask SPA backend (foodai.local:5000)
│   ├── config.py                     # Paths, thresholds, model config
│   ├── requirements.txt              # Pi-specific dependencies
│   ├── database/
│   │   └── db.py                     # SQLite schema: users, meals, meal_items
│   ├── pipeline/                     # Pi-adapted pipeline (same logic as root pipeline/)
│   │   ├── nutrition.py
│   │   ├── volume.py
│   │   ├── depth.py
│   │   ├── segmentation.py
│   │   └── camera.py
│   ├── models/
│   │   ├── best.pt                   # PyTorch fallback
│   │   └── best_ncnn_model/          # NCNN model (primary — runs on Pi without GPU)
│   └── static/
│       └── index.html                # Single-page frontend
│
├── data/
│   ├── nutrition_db.json             # Ghanaian food nutrition database
│   └── ...                           # Source CSVs, WAFCT data, GI references
│
├── models/
│   └── yolov8n-seg.pt                # Base model for training
│
├── requirements.txt                  # Development dependencies
└── README.md
```

---

## Quick Start

### Development (local)

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

### Raspberry Pi deployment

```bash
cd pi_deploy
pip install -r requirements.txt   # use CPU torch wheel on Pi
python app.py
```

Open `http://foodai.local:5000` from any device on the same network.

> **Pi camera:** Install via `sudo apt install -y python3-picamera2` — do not `pip install`.

---

## Supported Foods

| Food | Category | Portion Reference |
|------|----------|-------------------|
| Jollof Rice | Starch — spread | GDA: 180 g / serving |
| Waakye | Starch — spread | GDA: 170 g / serving |
| Plain Rice | Starch — spread | GDA: 180 g / serving |
| Fried Plantain | Starch — spread | GDA: 120 g / serving |
| Banku | Starch — moulded | Orange reference: 220 ml |
| Fufu | Starch — moulded | Orange reference: 220 ml |
| Rice Balls | Starch — moulded | Orange reference: 220 ml |
| Tuo Zaafi | Starch — moulded | Orange reference: 220 ml |
| Grilled Chicken | Protein | — |
| Tilapia | Protein | — |
| Fried Fish | Protein | — |
| Beans | Protein | — |
| Boiled Egg | Protein | — |
| Salad | Vegetable | — |
| Okro Soup | Soup / sauce | Excluded from plate ratios |
| Light Soup | Soup / sauce | Excluded from plate ratios |
| Shito | Soup / sauce | Excluded from plate ratios |

---

## Glycemic Load Rating

> GL = (GI / 100) × carbs_per_100g × (weight_g / 100)
>
> Source: Foster-Powell, Holt & Brand-Miller (2002)

| GL | Rating |
|----|--------|
| < 10 | 🟢 Low |
| 10 – 19 | 🟡 Medium |
| ≥ 20 | 🔴 High |

---

## Model Training & Export

**Train:**

```python
from ultralytics import YOLO

model = YOLO("models/yolov8n-seg.pt")
model.train(data="path/to/data.yaml", epochs=100, imgsz=640)
```

**Export to NCNN for Pi:**

```python
model = YOLO("runs/segment/train/weights/best.pt")
model.export(format="ncnn")
```

Copy the exported `best_ncnn_model/` folder to `pi_deploy/models/`.

---

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Intel MiDaS](https://github.com/isl-org/MiDaS)
- [Ghana Dietetic Association Serving Sizes — Lartey et al. 1999](https://www.gda.org.gh/)
- [International Tables of Glycemic Index — Foster-Powell et al. 2002](https://doi.org/10.1093/ajcn/76.1.5)
- [West African Food Composition Table (WAFCT) 2019](https://www.fao.org/infoods/infoods/tables-and-databases/africa/en/)

---

## License

MIT License — Educational / Research Use
