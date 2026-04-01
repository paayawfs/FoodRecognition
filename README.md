# Edge AI Food Identification System

## Ghanaian Food Recognition with Diabetic Dietary Recommendations

An edge-deployed AI system that photographs meals, identifies Ghanaian foods, estimates portion sizes, and provides diabetic-friendly dietary recommendations.

---

## 🎯 Features

- **Food Segmentation**: YOLOv8-seg detects and segments 10 Ghanaian foods
- **Depth Estimation**: MiDaS estimates food height for volume calculation
- **Portion Estimation**: Combines segmentation + depth → volume → weight
- **Nutrition Analysis**: Calculates calories, carbs, protein, and glycemic load
- **Diabetic Recommendations**: Rule-based dietary advice based on glycemic load

---

## 📁 Project Structure

```
capstone/
├── notebooks/                    # Jupyter notebooks (main implementation)
│   ├── 01_food_segmentation.ipynb    # YOLOv8-seg food detection
│   ├── 02_depth_estimation.ipynb     # MiDaS depth estimation
│   ├── 03_fusion_pipeline.ipynb      # Combine seg + depth → volume
│   ├── 04_nutrition_calculator.ipynb # Nutrition & GL calculations
│   └── 05_complete_demo.ipynb        # End-to-end demo
├── data/
│   └── nutrition_db.json         # Ghanaian food nutrition database
├── models/                       # Place trained models here
│   └── best.pt                   # (Your fine-tuned YOLOv8 model)
├── templates/
│   └── index.html                # Flask web interface
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd capstone
pip install -r requirements.txt
```

### 2. Run Jupyter Notebooks

Open notebooks in order to understand each component:

```bash
jupyter notebook notebooks/
```

### 3. Run Flask Web App

```bash
python app.py
```

Open http://localhost:5000 in your browser.

---

## 🍲 Supported Foods

| Food | Category | Glycemic Index |
|------|----------|----------------|
| Jollof Rice | Starch | 70 |
| Waakye | Starch | 55 |
| Banku | Starch | 65 |
| Fufu | Starch | 75 |
| Fried Plantain | Starch | 70 |
| Grilled Chicken | Protein | 0 |
| Grilled Tilapia | Protein | 0 |
| Kontomire Stew | Vegetable | 15 |
| Light Soup | Soup | 20 |
| Shito | Condiment | 10 |

---

## 🔧 Training Your Own Model

1. Annotate images in [Roboflow](https://roboflow.com) with polygon masks
2. Export dataset in YOLOv8 format
3. Train:

```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.train(data="path/to/data.yaml", epochs=100, imgsz=640)
```

4. Copy `runs/segment/train/weights/best.pt` to `models/best.pt`

---

## 📊 Glycemic Load Rating

| GL Range | Rating | Color |
|----------|--------|-------|
| 0-10 | Low | 🟢 Green |
| 11-19 | Medium | 🟡 Yellow |
| 20-30 | High | 🟠 Orange |
| 31+ | Very High | 🔴 Red |

---

## 🍓 Raspberry Pi Deployment

1. Flash Raspberry Pi OS (64-bit)
2. Install dependencies
3. Convert models to TFLite:

```python
model.export(format="tflite")
```

4. Run Flask app

---

## 📚 References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Intel MiDaS](https://github.com/isl-org/MiDaS)
- [FAO INFOODS](https://www.fao.org/infoods/)
- [USDA FoodData Central](https://fdc.nal.usda.gov/)

---

## 📝 License

MIT License - Educational/Research Use
"# FoodRecognition" 
