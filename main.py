import cv2
from analyzer import analyze_jump
from utils.visualization import save_and_show_output
from keypoints.data_loader import load_keypoints_data
from keypoints.yolo import get_key_points

video_path = "test_videos/test1.mp4"
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)

keypoints_data = load_keypoints_data(video_path, get_key_points, ignore_cache=True)

jump_data = analyze_jump(keypoints_data, fps)

save_and_show_output(video_path, video, jump_data,
                     keypoints_data, show=True, fps=fps)
