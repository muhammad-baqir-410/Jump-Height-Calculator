from utils.calculations import calculate_loss_and_params, calculate_jump_height, calculate_launch_velocity

LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12

def get_hip_keypoint_trajectories(keypoints_data, id):
    """Extract x and y coordinates for both left and right hips."""
    x_coords_left_hip = []
    y_coords_left_hip = []
    x_coords_right_hip = []
    y_coords_right_hip = []
    time_steps = sorted(keypoints_data[str(id)].keys(), key=int)    

    for frame in time_steps:
        keypoints = keypoints_data[str(id)][frame]["keypoints"]
        left_hip = [float(kp) for kp in keypoints[LEFT_HIP_INDEX][:2]]
        right_hip = [float(kp) for kp in keypoints[RIGHT_HIP_INDEX][:2]]
        x_coords_left_hip.append(left_hip[0])
        y_coords_left_hip.append(left_hip[1])
        x_coords_right_hip.append(right_hip[0])
        y_coords_right_hip.append(right_hip[1])
    time_steps = [int(time_step) for time_step in time_steps]
    
    return x_coords_left_hip, y_coords_left_hip, x_coords_right_hip, y_coords_right_hip, time_steps

def analyze_jump(keypoints_data):
    jump_data = {}
    for id in keypoints_data:
        try:
            x_coords_left_hip, y_coords_left_hip, x_coords_right_hip, y_coords_right_hip, time_steps = get_hip_keypoint_trajectories(keypoints_data, int(id))
            time_steps = [int(time_step) for time_step in time_steps]

            min_loss = float('inf')
            best_launch_frame = None
            best_landing_frame = None
            best_quadratic_params = None

            for launch_frame in range(1, max(time_steps) - 1):
                for landing_frame in range(launch_frame + 1, max(time_steps)):
                    current_loss, quadratic_params = calculate_loss_and_params([launch_frame, landing_frame], y_coords_left_hip, time_steps)
                    if current_loss < min_loss:
                        min_loss = current_loss
                        best_launch_frame = launch_frame
                        best_landing_frame = landing_frame
                        best_quadratic_params = quadratic_params

            midpoint_of_jump = (best_launch_frame + best_landing_frame) / 2
            vertex_of_curve = -best_quadratic_params[1] / (2 * best_quadratic_params[0])

            # Allow a margin of error around the midpoint
            margin_of_error = 5  # Define a suitable margin of error (in frames)
            direction_change_near_midpoint = abs(midpoint_of_jump - vertex_of_curve) <= margin_of_error
            concave_up = best_quadratic_params[0] > 0  # True if curve is concave up, False if concave down
            jumping = False
            jump_height = 0
            launch_velocity = 0
            if direction_change_near_midpoint and concave_up:
                jumping = True
                total_air_time = best_landing_frame - best_launch_frame
                fps = 30
                # calculate total air time in seconds as float
                total_air_time = total_air_time / fps
                print(f"Total air time: {total_air_time}")
                jump_height = calculate_jump_height(total_air_time)
                launch_velocity = calculate_launch_velocity(total_air_time)

            # print player id, jumping and launch and landing frames if jumping
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
