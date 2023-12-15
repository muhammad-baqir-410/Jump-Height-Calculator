from utils.calculations import calculate_loss_and_params, calculate_jump_height, calculate_launch_velocity
from scipy.ndimage import gaussian_filter

LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16

def get_limb_keypoint_trajectories(keypoints_data, id, LEFT_LIMB_INDEX=11, RIGHT_LIMB_INDEX=12):
    """Extract x and y coordinates for both left and right hips."""
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
    y_coords_left_hip = gaussian_filter(y_coords_left_hip, sigma=4)
    y_coords_right_hip = gaussian_filter(y_coords_right_hip, sigma=4)
    return x_coords_left_hip, y_coords_left_hip, x_coords_right_hip, y_coords_right_hip, time_steps

def analyze_jump(keypoints_data, fps=30):
    jump_data = {}
    for id in keypoints_data:
        try:
            x_coords_left_hip, y_coords_left_hip, x_coords_right_hip, y_coords_right_hip, time_steps = get_limb_keypoint_trajectories(keypoints_data, int(id))
            time_steps = [int(time_step) for time_step in time_steps]

            min_loss = float('inf')
            best_launch_frame = None
            best_landing_frame = None
            best_quadratic_params = None
            sorted_params = []
            for launch_frame in range(1, max(time_steps) - 10):
                for landing_frame in range(launch_frame + 10, max(time_steps)):
                    left_hip_loss, left_hip_quadratic_params = calculate_loss_and_params([launch_frame, landing_frame], y_coords_left_hip, time_steps)
                    right_hip_loss, right_hip_quadratic_params = calculate_loss_and_params([launch_frame, landing_frame], y_coords_right_hip, time_steps)
                    if left_hip_quadratic_params is None or right_hip_quadratic_params is None:
                        continue
                    current_loss = left_hip_loss + right_hip_loss

                    sorted_params.append((current_loss, (launch_frame, landing_frame, left_hip_quadratic_params, right_hip_quadratic_params)))

                    if current_loss < min_loss:
                        min_loss = current_loss
                        best_launch_frame = launch_frame
                        best_landing_frame = landing_frame
                        best_quadratic_params = [left_hip_quadratic_params, right_hip_quadratic_params]

            midpoint_of_jump = (best_launch_frame + best_landing_frame) / 2
            vertices_of_curves = [-quadratic_params[1] / (2 * quadratic_params[0]) for quadratic_params in best_quadratic_params if quadratic_params is not None]

            # Allow a margin of error around the midpoint
            margin_of_error = 5  
            direction_changes_near_midpoint = [abs(midpoint_of_jump - vertex_of_curve) <= margin_of_error for vertex_of_curve in vertices_of_curves]
            direction_change_near_midpoint = all(direction_changes_near_midpoint)
            concave_up = True
            for quadratic_params in best_quadratic_params:
                if quadratic_params is not None:
                    if quadratic_params[0] < 0:
                        concave_up = False
                        break
            jumping = False
            jump_height = 0
            launch_velocity = 0
            if direction_change_near_midpoint and concave_up:
                jumping = True
                best_launch_frame+=3
                total_air_time = best_landing_frame - best_launch_frame
                total_air_time = total_air_time / fps
                jump_height = calculate_jump_height(total_air_time)
                launch_velocity = calculate_launch_velocity(total_air_time)

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
        except Exception as e:
            print(f"Error processing player ID {id}: {e}")
    return jump_data
