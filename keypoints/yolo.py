from ultralytics import YOLO
from tqdm import tqdm

model = YOLO("yolov8n-pose.pt")

def get_key_points(path="test_videos/test1.mp4"):
    """
    The function `get_key_points` takes a video path as input and returns a dictionary containing
    keypoints data for each frame and each object ID in the video.
    
    :param path: The `path` parameter is the path to the video file from which you want to extract
    keypoints data. By default, it is set to "test_videos/test1.mp4", defaults to test_videos/test1.mp4
    (optional)
    :return: a dictionary containing keypoints data for each id in each frame of the video.
    """
    
    print("***Getting keypoints data***")
    results = model.track(source = path, save=False, show=False, tracker="botsort.yaml", stream=True, verbose=False)

    # dictionary to store keypoints data against each id for all frames
    keypoints_data = {}
    frame_count = 0
    for result in tqdm(results):
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