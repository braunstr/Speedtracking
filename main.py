import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from IPython.display import HTML, display
from base64 import b64encode

SOURCE = np.array([[740, 430], [1200, 400], [2100, 830], [-310, 870]])

TARGET_WIDTH = 35
TARGET_HEIGHT = 190
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH-1, 0],
    [TARGET_WIDTH-1, TARGET_HEIGHT-1],
    [0, TARGET_HEIGHT-1],
])


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Inference and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO("yolov8x.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Thinner bounding box and smaller text scale
    thickness = 1
    text_scale = 0.5

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Set up VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('result.mp4', fourcc, video_info.fps, video_info.resolution_wh)

    for frame in frame_generator:

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        labels = []
        bbox_colors = []
        annotated_frame = frame.copy()

        for tracker_id, [_, y], bbox in zip(detections.tracker_id, points, detections.xyxy):
            coordinates[tracker_id].append(y)

            if len(coordinates[tracker_id]) < video_info.fps/2:
                labels.append(f"#{tracker_id}")
                bbox_color = (255, 255, 255)  # White
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"{int(speed)} km/h")  # #{tracker_id} can be put after f

                if speed < 100:
                    bbox_color = (0, 255, 0)  # Green
                elif 100 <= speed < 140:
                    bbox_color = (0, 255, 255)  # Yellow
                else:
                    bbox_color = (0, 0, 255)  # Red

            bbox_colors.append(bbox_color)
            annotated_frame = cv2.rectangle(
                annotated_frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                bbox_color,
                thickness
            )

        for label, bbox_color, bbox in zip(labels, bbox_colors, detections.xyxy):
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
            text_width, text_height = text_size
            text_x, text_y = int(bbox[0]), int(bbox[3]) + text_height + 5

            if text_y + text_height > frame.shape[0]:
                text_y = int(bbox[3]) - 5

            cv2.rectangle(
                annotated_frame,
                (text_x, text_y - text_height - 5),
                (text_x + text_width, text_y + 5),
                bbox_color,
                cv2.FILLED
            )
            cv2.putText(
                annotated_frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (0, 0, 0),  # Black text
                thickness
            )

        # Write the annotated frame to the video
        out.write(annotated_frame)

    # Release the video writer
    out.release()
    cv2.destroyAllWindows()

    # Display the saved video
    video_path = 'result.mp4'
    mp4 = open(video_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    display(HTML("""
        <video width=400 controls>
            <source src="%s" type="video/mp4">
        </video>
    """ % data_url))
