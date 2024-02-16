from utils.calculations import calculate_jump_height, calculate_launch_velocity, find_parabolic_curve
import numpy as np

LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16


def get_limb_keypoint_trajectories(keypoints_data, id, LEFT_LIMB_INDEX=11, RIGHT_LIMB_INDEX=12):
    """
    The function `get_limb_keypoint_trajectories` extracts x and y coordinates for both left and right
    hips from a given keypoints data.

    :param keypoints_data: The `keypoints_data` parameter is a dictionary that contains the keypoints
    data for different frames and different individuals. The keys of the dictionary represent the
    individual IDs, and the values are dictionaries that contain the keypoints data for each frame
    :param id: The `id` parameter is used to specify the ID of the person for whom we want to extract
    the limb keypoints trajectories
    :param LEFT_LIMB_INDEX: The LEFT_LIMB_INDEX parameter is the index of the left hip in the keypoints
    list. It is used to extract the x and y coordinates of the left hip from the keypoints data,
    defaults to 11 (optional)
    :param RIGHT_LIMB_INDEX: The `RIGHT_LIMB_INDEX` parameter is the index of the right hip in the
    keypoints list. It is used to extract the x and y coordinates of the right hip from the keypoints
    data, defaults to 12 (optional)
    :return: five lists: x_coords_left_hip, y_coords_left_hip, x_coords_right_hip, y_coords_right_hip,
    and time_steps.
    """

    x_coords_left_hip = []
    y_coords_left_hip = []
    x_coords_right_hip = []
    y_coords_right_hip = []
    time_steps = sorted(keypoints_data[str(id)].keys(), key=int)

    for frame in time_steps:
        keypoints = keypoints_data[str(id)][frame]["keypoints"]
        left_hip = [float(kp) for kp in keypoints[LEFT_LIMB_INDEX][:2]]
        right_hip = [float(kp) for kp in keypoints[RIGHT_LIMB_INDEX][:2]]
        x_coords_left_hip.append(left_hip[0])
        y_coords_left_hip.append(left_hip[1])
        x_coords_right_hip.append(right_hip[0])
        y_coords_right_hip.append(right_hip[1])
    time_steps = [int(time_step) for time_step in time_steps]
    return x_coords_left_hip, y_coords_left_hip, x_coords_right_hip, y_coords_right_hip, time_steps


def analyze_jump(keypoints_data, fps=30):
    """
    The `analyze_jump` function takes in keypoints data of players' hip positions over time and
    calculates various jump-related metrics such as launch frame, landing frame, jump height, and launch
    velocity for each player.

    :param keypoints_data: The `keypoints_data` parameter is a dictionary that contains the keypoint
    data for each player. Each player is identified by their ID, and the keypoint data for each player
    includes the x and y coordinates of the left hip and right hip keypoints, as well as the time steps
    :param fps: The parameter `fps` stands for frames per second and represents the frame rate of the
    video or animation being analyzed. It is used to calculate the total air time of the jump and the
    launch velocity, defaults to 30 (optional)
    :return: The function `analyze_jump` returns a dictionary `jump_data` which contains information
    about the jump analysis for each player ID. The keys of the dictionary are the player IDs, and the
    values are dictionaries containing the following information:
    """
    jump_data = {}
    for id in keypoints_data:
        x_coords_left_hip, y_coords_left_hip, x_coords_right_hip, y_coords_right_hip, time_steps = get_limb_keypoint_trajectories(
            keypoints_data, int(id))
        y_coords_left_hip = np.array(y_coords_left_hip)
        best_launch_frame = None
        best_landing_frame = None
        jump_height = 0
        launch_velocity = 0
        jumping = False
        threshold = 0.5
        window_size = 10
        try:
            best_launch_frame, best_landing_frame = find_parabolic_curve(
                y_coords_left_hip, window_size=window_size, threshold=threshold, find_minimum_peak=True)
        except Exception as e:
            print(f"Error processing player ID {id}: {e}")
            best_launch_frame = None
            best_landing_frame = None

        if best_launch_frame and best_landing_frame:
            best_launch_frame += 3
            best_landing_frame += 1
            jumping = True
            total_air_time = best_landing_frame - best_launch_frame
            total_air_time = total_air_time / fps
            jump_height = calculate_jump_height(total_air_time)
            launch_velocity = calculate_launch_velocity(total_air_time)
        else:
            jumping = False

        jump_data[id] = {
            "jumping": jumping,
            "launch_frame": best_launch_frame,
            "landing_frame": best_landing_frame,
            "jump_height": jump_height,
            "launch_velocity": launch_velocity
        }
        if jumping:
            print(f"For player ID {id}: Launch frame: {best_launch_frame}, Landing frame: {best_landing_frame}, Jumping: {jumping}, Jump height: {jump_height}, Launch velocity: {launch_velocity}")
        else:
            print(f"For player ID {id}: Not jumping")
    return jump_data
