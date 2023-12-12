import numpy as np
import warnings
from numpy import RankWarning

GRAVITY = 9.81

def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def calculate_loss_and_params(params, y_coords, time_steps):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RankWarning)
            y_coords_array = np.array(y_coords)
            time_steps_array = np.array(time_steps)

            launch_frame = int(np.round(params[0]))
            landing_frame = int(np.round(params[1]))

            launch_frame = max(1, min(launch_frame, len(time_steps_array) - 1))
            landing_frame = max(launch_frame + 1, min(landing_frame, len(time_steps_array)))

            pre_jump = y_coords_array[time_steps_array <= launch_frame]
            during_jump = y_coords_array[(time_steps_array > launch_frame) & (time_steps_array <= landing_frame)]
            post_jump = y_coords_array[time_steps_array > landing_frame]

            time_pre_jump = time_steps_array[time_steps_array <= launch_frame]
            time_during_jump = time_steps_array[(time_steps_array > launch_frame) & (time_steps_array <= landing_frame)]
            time_post_jump = time_steps_array[time_steps_array > landing_frame]

            linear_params_pre = np.polyfit(time_pre_jump, pre_jump, 1)
            linear_params_post = np.polyfit(time_post_jump, post_jump, 1)
            quadratic_params = np.polyfit(time_during_jump, during_jump, 2)

            loss = np.sum((linear_model(time_pre_jump, *linear_params_pre) - pre_jump) ** 2)
            loss += np.sum((quadratic_model(time_during_jump, *quadratic_params) - during_jump) ** 2)
            loss += np.sum((linear_model(time_post_jump, *linear_params_post) - post_jump) ** 2)

            return loss, quadratic_params
    except Exception as e:
        return float('inf'), None

def calculate_jump_height(total_air_time):
    jump_height = GRAVITY * total_air_time**2 / 8
    jump_height = jump_height * 100
    return jump_height

def calculate_launch_velocity(total_air_time):
    launch_velocity = GRAVITY * total_air_time / 2
    return launch_velocity
