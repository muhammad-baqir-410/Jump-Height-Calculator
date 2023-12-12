import cv2

def show_jump_info_on_video(video_path, jump_data, keypoints_data):
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path.replace(".mp4", "_output.mp4"), fourcc, 30, (int(video.get(3)), int(video.get(4))))
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        for id in jump_data:
            if jump_data[id]["jumping"] and frame_count >= jump_data[id]["launch_frame"] and frame_count <= jump_data[id]["landing_frame"]:
                text = f"ID: {id}, Jump height: {jump_data[id]['jump_height']:.2f} cm, Launch velocity: {jump_data[id]['launch_velocity']:.2f} m/s"
                box = keypoints_data[str(id)][str(frame_count)]["box"]
                box = [int(b) for b in box]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Frame", frame)
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()