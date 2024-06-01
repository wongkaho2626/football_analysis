import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("./models/best.pt")
tracker = sv.ByteTrack()

ellipse_annotator = sv.EllipseAnnotator()
label_annotator = sv.LabelAnnotator()


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    ellipse_annotator = sv.EllipseAnnotator()

    detections = tracker.update_with_detections(detections)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated_frame = ellipse_annotator.annotate(
        scene=frame.copy(), detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )
    return annotated_frame


sv.process_video(
    source_path="input_videos/08fd33_4.mp4",
    target_path="output_videos/byteTrack.avi",
    callback=callback,
)
