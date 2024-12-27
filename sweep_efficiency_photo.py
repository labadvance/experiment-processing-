import os
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
    """Processes the image by converting to grayscale, normalizing, and applying adaptive thresholding."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    binary_img = cv2.adaptiveThreshold(normalized_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, c)
    return binary_img

def calculate_sweep_efficiency(binary_img, binary_ref):
    """Calculates sweep efficiency based on binary images."""
    return (1 - count_pixels(binary_img, 255) / count_pixels(binary_ref, 255)) * 100

def save_image(output_dir, filename, image):
    """Saves an image to the specified output directory."""
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)

# =============================
# Visualization Function
# =============================

def visualize_images(start_image, end_image, binary_start, binary_end, sweep_efficiency_binary, start_params, end_params):
    """Displays and annotates the original and binary images with Sweep Efficiency."""
    plt.figure(figsize=(8, 6))

    # Binarized start image
    plt.subplot(2, 2, 1)
    plt.imshow(binary_start, cmap='gray')
    plt.title(f'Binarized Start (blockSize={start_params[0]}, c={start_params[1]})')
    plt.axis('off')

    # Binarized end image
    plt.subplot(2, 2, 2)
    plt.imshow(binary_end, cmap='gray')
    plt.title(f'Binarized End (blockSize={end_params[0]}, c={end_params[1]})')
    plt.axis('off')

    # Original start image
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(start_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Start Image')
    plt.axis('off')

    # Original end image
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(end_image, cv2.COLOR_BGR2RGB))
    plt.title('Original End Image')
    plt.axis('off')

    # Sweep Efficiency Title
    plt.suptitle(f'Sweep Efficiency (Binary): {sweep_efficiency_binary:.2f}%', fontsize=14)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# =============================
# Main Execution
# =============================

if __name__ == "__main__":
    # Configuration
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)

    start_image_path = 'до.png'
    end_image_path = 'после.png'

    start_block_sizes = [91]
    start_c_values = [-11]
    end_block_sizes = [41]
    end_c_values = [-22]

    # Load Images
    start_image = cv2.imread(start_image_path)
    end_image = cv2.imread(end_image_path)

    if start_image is None or end_image is None:
        print("Error loading images. Check file paths.")
        exit()

    # Iterate through parameter combinations
    for start_blockSize in start_block_sizes:
        for start_c in start_c_values:
            for end_blockSize in end_block_sizes:
                for end_c in end_c_values:
                    print(f"Parameters: start_blockSize={start_blockSize}, start_c={start_c}, end_blockSize={end_blockSize}, end_c={end_c}")

                    # Process Images
                    binary_start = process_image(start_image, start_blockSize, start_c)
                    binary_end = process_image(end_image, end_blockSize, end_c)

                    # Calculate Sweep Efficiency
                    sweep_efficiency_binary = calculate_sweep_efficiency(binary_end, binary_start)

                    # Save Images
                    save_image(output_dir, 'binary_start.png', binary_start)
                    save_image(output_dir, 'binary_end.png', binary_end)
                    save_image(output_dir, 'original_start.png', start_image)
                    save_image(output_dir, 'original_end.png', end_image)

                    # Visualize Results
                    visualize_images(start_image, end_image, binary_start, binary_end, sweep_efficiency_binary, 
                                     (start_blockSize, start_c), (end_blockSize, end_c))

                    # User Input to Exit
                    exit_input = input("Enter 'exit' to stop or press Enter to continue: ")
                    if exit_input.lower() == 'exit':
                        exit()
