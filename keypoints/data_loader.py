import json
import os

def load_keypoints_data(video_path, get_key_points_function, ignore_cache=False):
    """
    The function `load_keypoints_data` loads keypoints data from a cache file or generates it using a
    provided function if the cache file does not exist or if the `ignore_cache` flag is set to `True`.
    
    :param video_path: The path to the video file from which you want to extract keypoints
    :param get_key_points_function: The `get_key_points_function` is a function that takes the path of a
    video file as input and returns the key points data for that video. It is used to extract the key
    points data from the video
    :param ignore_cache: The `ignore_cache` parameter is a boolean flag that determines whether to
    ignore the cached keypoints data or not. If `ignore_cache` is set to `True`, the function will
    always compute the keypoints data and overwrite the existing cache. If `ignore_cache` is set to
    `False` (default, defaults to False (optional)
    :return: the keypoints data loaded from the cache file.
    """
    
    keyppoints_cache_dir = "keypoints_cache"
    video_file_name = video_path.split("/")[-1]
    json_file_name = video_file_name.replace(".mp4", ".json")
    keypoints_cache_file_path = os.path.join(
        keyppoints_cache_dir, json_file_name)
    
    if not os.path.exists(keypoints_cache_file_path) or ignore_cache:
        keypoints_data = get_key_points_function(path=video_path)
        with open(keypoints_cache_file_path, "w") as f:
            json.dump(keypoints_data, f)

    with open(keypoints_cache_file_path, "r") as f:
        return json.load(f)
