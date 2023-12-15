from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

def get_key_points(path=""):
    print("***Processing Video***")
    results = model.track(source = path, save=False, show=False, tracker="bytetrack.yaml", stream=True, verbose=False)

    # dictionary to store keypoints data against each id for all frames
    keypoints_data = {}
    frame_count = 0
    for result in results:
        frame_count += 1
        ids = result.boxes.id.cpu().numpy().tolist() if result.boxes.id is not None else []
        ids = [int(id) for id in ids]
        boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes.xyxy is not None else []
        keypoints = result.keypoints.data.cpu().numpy().tolist() if result.keypoints.data is not None else []
        for i, id in enumerate(ids):
            if id not in keypoints_data:
                keypoints_data[id] = {}
            keypoints_data[id][frame_count] = {
                "box": boxes[i],
                "keypoints": keypoints[i]
            }

    return keypoints_data