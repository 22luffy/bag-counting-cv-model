import cv2
import os
from ultralytics import YOLO

# Load your trained bag model
model = YOLO("yolov8n.pt")  # replace with your custom bag model if different

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "outputs"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    video_name = os.path.basename(video_path)
    output_path = os.path.join(OUTPUT_FOLDER, "processed_" + video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    line_x = width // 2   # vertical counting line in middle
    counted_ids = set()
    total_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # If crosses line and not counted
                if center_x > line_x and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1

        # Draw counting line
        cv2.line(frame, (line_x, 0), (line_x, height), (255,0,0), 3)

        # Show count
        cv2.putText(frame, f"Bag Count: {total_count}", 
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0,0,255),
                    3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Finished processing {video_name}")

def process_all_videos():
    for file in os.listdir(VIDEO_FOLDER):
        if file.endswith(".mp4") or file.endswith(".avi"):
            process_video(os.path.join(VIDEO_FOLDER, file))

if __name__ == "__main__":
    process_all_videos()