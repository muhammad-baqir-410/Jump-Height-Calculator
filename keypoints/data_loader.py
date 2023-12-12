import json
import os

def load_keypoints_data(video_path, get_key_points_function):

    if not os.path.exists(video_path.replace(".mp4", ".json")):
        keypoints_data = get_key_points_function(path=video_path)
        with open(video_path.replace(".mp4", ".json"), "w") as f:
            json.dump(keypoints_data, f)
    
    with open(video_path.replace(".mp4", ".json"), "r") as f:
        return json.load(f)