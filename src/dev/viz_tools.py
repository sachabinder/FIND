import init_paths
import cv2

import numpy as np


def add_image_title(image, title, padding=20, text_height=50):
    """
    Add a title to an image by placing it on top of a white background.

    :param image: The image to add the title to.
    :param title: The title to add to the image.
    :param padding: The padding between the image and the background.
    """

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the dimensions for the background based on the image and padding
    background_height = height + 2 * padding + text_height
    background_width = max(width, text_height) + 2 * padding

    # Create a white background with the calculated dimensions
    background = np.ones((background_height, background_width, 3), dtype=np.uint8) * 255

    # Calculate the coordinates to place the image within the background
    x = (background_width - width) // 2
    y = padding

    # Place the image on the background
    background[y : y + height, x : x + width] = image * 255

    # Add black text on top of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_thickness = 1
    text_size = cv2.getTextSize(title, font, font_scale, text_thickness)[0]
    text_x = (background_width - text_size[0]) // 2
    text_y = y + height + padding + text_size[1] + 5
    text_origin = (text_x, text_y)
    text_color = (0, 0, 0)  # Black color

    cv2.putText(
        background,
        title,
        text_origin,
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )

    return background


def create_combined_image(image1, image2):
    """
    Combine two images side by side into a single image.

    :param image1: The first image to combine.
    :param image2: The second image to combine.
    """
    # Resize the images to have the same height
    height = max(image1.shape[0], image2.shape[0])
    width1 = int(image1.shape[1] * (height / image1.shape[0]))
    width2 = int(image2.shape[1] * (height / image2.shape[0]))
    resized_image1 = cv2.resize(image1, (width1, height))
    resized_image2 = cv2.resize(image2, (width2, height))

    # Create a black canvas to place the combined image
    combined_width = width1 + width2
    combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)

    # Place the resized images side by side on the canvas
    combined_image[:, :width1] = resized_image1
    combined_image[:, width1:] = resized_image2

    return combined_image


def create_grayscale_video(images, output_path, fps):
    """
    Create a grayscale video from a list of images.

    :param images: The list of images to create the video from.
    :param output_path: The path to save the video to.
    :param fps: The frames per second of the video.
    """
    # Get the shape of the first image in the list
    height, width, _ = images[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can choose other codecs as well
    video_writer = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height), isColor=False
    )

    # Iterate over the list of images and write them to the video
    for image in images:
        # Write the BGR image to the video
        video_writer.write(image)

    # Release the video writer
    video_writer.release()
