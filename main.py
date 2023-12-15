import cv2
from analyzer import analyze_jump
from utils.visualization import show_jump_info_on_video
from keypoints.data_loader import load_keypoints_data
from keypoints.yolo import get_key_points

video_path = "videos/jump2.mp4"
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)

keypoints_data = load_keypoints_data(video_path, get_key_points)

jump_data = analyze_jump(keypoints_data, fps)

show_jump_info_on_video(video_path, video, jump_data, keypoints_data, fps)