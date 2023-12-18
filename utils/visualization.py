import cv2
import numpy as np

FONT_SCALE = 0.9
FONT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

def show_jump_info_on_video(video_path, video, jump_data, keypoints_data, fps=30):
    frame_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path.replace(".mp4", "_output.mp4"), fourcc, fps, (int(video.get(3)), int(video.get(4))))

    # Define colors for each keypoint
    colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255), # Yellow
        (128, 128, 128), # Gray
        (128, 0, 0),   # Maroon
        (0, 128, 0),   # Dark Green
        (0, 0, 128),   # Navy
        (128, 128, 0), # Olive
        (128, 0, 128), # Purple
        (0, 128, 128), # Teal
        (192, 192, 192), # Silver
        (64, 0, 0),    # Dark Red
        (0, 64, 0),    # Darker Green
        (0, 0, 64)     # Dark Blue
    ]
    # Define skeleton connections
    skeleton = [
        (0, 1), (1, 2), (0, 3), (3, 4), 
        (5, 6), (5, 7), (7, 9), (6, 8), 
        (8, 10), (5, 11), (6, 12), (11, 13), 
        (13, 15), (12, 14), (14, 16), (11, 12)
    ]

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        current_time = frame_count / fps

        for id in jump_data:
            if jump_data[id]["jumping"]:
                t_0 = jump_data[id]["launch_frame"] / fps
                v_0 = jump_data[id]["launch_velocity"]
                g = 9.81  # Gravity constant

                if frame_count >= jump_data[id]["launch_frame"] and frame_count <= jump_data[id]["landing_frame"]:
                    h = (v_0 * (current_time - t_0) - 0.5 * g * (current_time - t_0) ** 2)*100     
                    text = f"ID: {id}\nHeight: {h:.2f} cm\nLaunch speed: {v_0:.2f} m/s"
                    if str(frame_count) in keypoints_data[str(id)]:
                        box = keypoints_data[str(id)][str(frame_count)]["box"]
                        keypoints = keypoints_data[str(id)][str(frame_count)]["keypoints"]
                        box = [int(b) for b in box]
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

                        box_center_y = (box[1] + box[3]) // 2
                        lines = text.split('\n')
                        line_spacing = 5
                        text_width, text_height = 0, 0

                        for line in lines:
                            text_size = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                            text_width = max(text_width, text_size[0])
                            text_height += text_size[1] + line_spacing
                        text_height -= line_spacing

                        text_start_x = box[2] + 10
                        text_start_y = box_center_y - text_height // 2
                        for i, line in enumerate(lines):
                            text_size = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                            cv2.putText(frame, line, (text_start_x, text_start_y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
                            text_start_y += line_spacing + text_size[1]

                        # Draw keypoints with different colors
                        for i, kp in enumerate(keypoints):
                            color = colors[i % len(colors)]  # Cycle through colors
                            cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)

                        # Draw lines for skeleton
                        for start, end in skeleton:
                            if start < len(keypoints) and end < len(keypoints):
                                start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
                                end_point = (int(keypoints[end][0]), int(keypoints[end][1]))
                                cv2.line(frame, start_point, end_point, colors[len(skeleton) % len(colors)], 2)  # Use a different color for skeleton lines
                if frame_count > jump_data[id]["landing_frame"]:
                    h = jump_data[id]["jump_height"]
                    text = f"ID: {id}, Max Height: {h:.2f} cm, Launch velocity: {v_0:.2f} m/s \n"
                    # put multi-line text on top of the frame by splitting the text on newline character
                    lines = text.split('\n')
                    line_spacing = 5
                    text_width, text_height = 0, 0
                    for line in lines:
                        text_size = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                        text_width = max(text_width, text_size[0])
                        text_height += text_size[1] + line_spacing
                    text_height -= line_spacing
                    text_start_x = 10
                    text_start_y = 30
                    for i, line in enumerate(lines):
                        text_size = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)[0]
                        cv2.putText(frame, line, (text_start_x, text_start_y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
                        text_start_y += line_spacing + text_size[1]


                        
        cv2.imshow("Frame", frame)
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()
