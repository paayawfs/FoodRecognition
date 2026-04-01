from ultralytics import YOLO


class FoodSegmenter:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = (
            list(self.model.names.values())
            if hasattr(self.model, 'names') else []
        )
        print(f"Segmenter loaded. Classes ({len(self.class_names)}): {self.class_names}")

    def predict(self, image_bgr, confidence=0.40):
        """Run inference. Returns Ultralytics Results object."""
        return self.model.predict(image_bgr, conf=confidence, verbose=False)[0]
