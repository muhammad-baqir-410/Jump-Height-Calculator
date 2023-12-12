from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

def get_key_points(path=""):
    results = model.track(source = path, save=False, show=True, tracker="bytetrack.yaml", stream=True, verbose=False)

    # dictionary to store keypoints data against each id for all frames
    keypoints_data = {}
    frame_count = 0
    for result in results:
        frame_count += 1
        ids = result.boxes.id.cpu().numpy().tolist()
        ids = [int(id) for id in ids]
        # print(f"Frame: {frame_count}, IDs: {ids}")
        boxes = result.boxes.xyxy.cpu().numpy().tolist() # boxes are in xyxy format
        keypoints = result.keypoints.data.cpu().numpy().tolist() # keypoints are in xyconf format
        for i, id in enumerate(ids):
            if id not in keypoints_data:
                keypoints_data[id] = {}
            keypoints_data[id][frame_count] = {
                "box": boxes[i],
                "keypoints": keypoints[i]
            }

    return keypoints_data