import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Utility Functions
# =============================

def count_pixels(image, pixel_value):
    """Counts the number of pixels with a specific value in the image."""
    return np.sum(image == pixel_value)

def process_image(image, blockSize, c):
    """Processes the image by extracting the green channel, normalizing it, and applying adaptive thresholding."""
    green_channel = image[:, :, 1]
    normalized_img = cv2.normalize(green_channel, None, 0, 255, cv2.NORM_MINMAX)
    binary_img = cv2.adaptiveThreshold(normalized_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, c)
    return binary_img

def calculate_sweep_efficiency(binary_img, binary_ref):
    """Calculates sweep efficiency as a percentage."""
    efficiency = (1 - count_pixels(binary_img, 255) / count_pixels(binary_ref, 255)) * 100
    return max(efficiency if efficiency >= 2 else 0, 0)

# =============================
# Video Processing
# =============================

def process_video(video_path, output_video_path, output_binary_path, blockSize, c, Q, volume_cubic_micrometres):
    """Processes a video to calculate and annotate sweep efficiency over time."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_height = frame_height + 50
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, output_height), isColor=True)
    out_binary = cv2.VideoWriter(output_binary_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Reference frame (1 second mark)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(0 * fps))
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Error reading reference frame.")

    ref_frame_green = ref_frame.copy()
    ref_frame_green[:, :, [0, 2]] = 0
    binary_ref = process_image(ref_frame_green, blockSize, c)

    # Variables for plotting
    sweep_efficiency_values = []
    frame_times = []
    pore_volume_ratios = []
    frame_number = 0
    volume_ml = volume_cubic_micrometres / 1e12

    # Process video frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1 * fps))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        green_frame = frame.copy()
        green_frame[:, :, [0, 2]] = 0

        # Equalize brightness
        green_channel = green_frame[:, :, 1]
        rows, cols = green_channel.shape
        gradient = np.tile(np.linspace(0.5, 1.15, cols), (rows, 1))
        equalized_green = (green_channel * gradient).astype(np.uint8)
        green_frame[:, :, 1] = cv2.normalize(equalized_green, None, 0, 255, cv2.NORM_MINMAX)

        binary_frame = process_image(green_frame, blockSize, c)
        sweep_efficiency = calculate_sweep_efficiency(binary_frame, binary_ref)

        # Calculate metrics
        frame_time = frame_number / fps
        pore_volume_ratio = (Q * frame_time / 60) / volume_ml

        frame_times.append(frame_time)
        sweep_efficiency_values.append(sweep_efficiency)
        pore_volume_ratios.append(pore_volume_ratio)
        frame_number += 1

        # Annotate frame
        output_frame = np.zeros((output_height, frame_width, 3), dtype=np.uint8)
        output_frame[:frame_height, :] = green_frame

        text = f'Sweep Efficiency: {sweep_efficiency:.2f}% | Pore Volume Ratio: {pore_volume_ratio:.2f} V/Vp'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = frame_height + 35
        cv2.putText(output_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(output_frame)
        out_binary.write(binary_frame)

    cap.release()
    out.release()
    out_binary.release()

    return pore_volume_ratios, sweep_efficiency_values, frame_times

# =============================
# Plotting Results
# =============================

def plot_results(pore_volume_ratios, sweep_efficiency_values, frame_times):
    """Plots sweep efficiency versus pore volume ratio with a secondary axis for time."""
    fig, ax = plt.subplots()
    ax.plot(pore_volume_ratios, sweep_efficiency_values, label='Sweep Efficiency', color='b')
    ax.set_xlabel('Pore Volume Ratio (V/Vp)')
    ax.set_ylabel('Sweep Efficiency (%)')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.set_xlabel('Time (s)')
    ax_time.set_xticks(pore_volume_ratios[::len(pore_volume_ratios)//10])
    ax_time.set_xticklabels([f'{t:.0f}' for t in frame_times[::len(frame_times)//10]])

    plt.tight_layout()
    plt.show()

# =============================
# Main Execution
# =============================

if __name__ == "__main__":
    video_path = 'LTG_4.mp4'
    output_video_path = 'sweep_LTG.mp4'
    output_binary_path = 'binary_sweep_LTG.mp4'

    # Parameters
    blockSize = 51
    c = -7
    Q = 0.0001  # Flow rate in ml/min
    volume_cubic_micrometres = 14362451

    # Process video and plot results
    pore_volume_ratios, sweep_efficiency_values, frame_times = process_video(
        video_path, output_video_path, output_binary_path, blockSize, c, Q, volume_cubic_micrometres
    )
    plot_results(pore_volume_ratios, sweep_efficiency_values, frame_times)
