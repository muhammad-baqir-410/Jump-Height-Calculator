import numpy as np
import warnings
from numpy import RankWarning
from scipy.signal import find_peaks
from numpy.polynomial.polynomial import Polynomial

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
            # replace y_coords_array with a smoothed version of itself
            time_steps_array = np.array(time_steps)

            launch_frame = int(np.round(params[0]))
            landing_frame = int(np.round(params[1]))

            launch_frame = max(1, min(launch_frame, len(time_steps_array) - 1))
            landing_frame = max(launch_frame + 1, min(landing_frame, len(time_steps_array)))

            pre_jump = y_coords_array[time_steps_array <= launch_frame]
            during_jump = y_coords_array[(time_steps_array > launch_frame) & (time_steps_array <= landing_frame)]
            post_jump = y_coords_array[time_steps_array > landing_frame]
            if pre_jump.size == 0 or during_jump.size == 0 or post_jump.size == 0:
                return float('inf'), None
            # remove outliers from pre_jump, during_jump and post_jump
            pre_jump_mean = np.mean(pre_jump)
            pre_jump_std = np.std(pre_jump)
            pre_jump = pre_jump[abs(pre_jump - pre_jump_mean) < 3 * pre_jump_std]

            during_jump_mean = np.mean(during_jump)
            during_jump_std = np.std(during_jump)
            during_jump = during_jump[abs(during_jump - during_jump_mean) < 3 * during_jump_std]

            post_jump_mean = np.mean(post_jump)
            post_jump_std = np.std(post_jump)
            post_jump = post_jump[abs(post_jump - post_jump_mean) < 3 * post_jump_std]

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
    
def find_parabolic_curve(signal, window_size=20, threshold=1, find_minimum_peak=True):
    """
    Find the start and end points of a concave-up parabolic curve in a signal.

    Parameters:
    signal (numpy array): The signal in which to find the parabolic curve.
    window_size (int): The size of the window around the peak or valley for curve fitting.
    threshold (float): The threshold for deviation from the fitted curve to determine the bounds.
    find_minimum_peak (bool): If True, finds the curve around the minimum peak (valley); otherwise, the maximum peak.

    Returns:
    tuple: Start and end points of the parabolic curve.
    """
    try:
        # Validate input
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array.")

        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        if threshold <= 0:
            raise ValueError("Threshold must be a positive number.")

        # Invert the signal if finding minimum peak (valley)
        processed_signal = -signal if find_minimum_peak else signal

        # Find peaks or valleys based on the processed signal
        peaks, _ = find_peaks(processed_signal, prominence=30)
        if len(peaks) == 0:
            return None, None

        # Choose the most prominent peak or valley
        target_peak = peaks[np.argmin(signal[peaks])] if find_minimum_peak else peaks[np.argmax(signal[peaks])]

        # Define the window for curve fitting
        start = max(target_peak - window_size, 0)
        end = min(target_peak + window_size, len(signal))

        # Fit a parabola in the window
        x = np.arange(start, end)
        y = signal[start:end]
        p = Polynomial.fit(x, y, 2)

        # Find the start and end points of the parabola
        deviation = np.abs(p(x) - y)
        indices_over_threshold = np.where(deviation > threshold)[0]

        if len(indices_over_threshold) > 0:
            start_of_parabola = indices_over_threshold[0] + start
            end_of_parabola = indices_over_threshold[-1] + start
        else:
            start_of_parabola, end_of_parabola = start, end

        return start_of_parabola, end_of_parabola

    except ValueError as e:
        # Handle the value errors and return a message
        return f"Error: {str(e)}"
    except Exception as e:
        # Handle any other exceptions
        return f"An unexpected error occurred: {str(e)}"



def calculate_jump_height(total_air_time):
    jump_height = GRAVITY * total_air_time**2 / 8
    jump_height = jump_height * 100
    return jump_height

def calculate_launch_velocity(total_air_time):
    launch_velocity = GRAVITY * total_air_time / 2
    return launch_velocity
