import json
from pathlib import Path
from ultralytics import YOLO


class FoodSegmenter:
    def __init__(self, model_path, class_names_path=None):
        self.model = YOLO(model_path)

        # NCNN models may not populate model.names — fall back to class_names.json
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = list(self.model.names.values())
        elif class_names_path and Path(class_names_path).exists():
            raw = json.loads(Path(class_names_path).read_text(encoding='utf-8'))
            # {0: "Banku", ...} or {"0": "Banku", ...}
            self.class_names = [raw[str(i)] for i in range(len(raw))]
        else:
            self.class_names = []

        print(f"Segmenter loaded. Classes ({len(self.class_names)}): {self.class_names}")

    def predict(self, image_bgr, confidence=0.40):
        """Run inference. Returns Ultralytics Results object."""
        return self.model.predict(image_bgr, conf=confidence, verbose=False)[0]
