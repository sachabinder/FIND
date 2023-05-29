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
    background[y : y + height, x : x + width] = (
        image if image.dtype == np.uint8 else image * 255
    )

    # Add black text on top of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
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


def create_video_from_images(image_list, output_path, fps):
    # Get the dimensions of the first image in the list
    height, width, _ = image_list[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can choose other codecs as well
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate over the image list and write each frame to the video
    for image in image_list:
        video_writer.write(image)

    # Release the video writer
    video_writer.release()


def overlay_images(gt_img, pred_img):
    """
    Overlay the ground truth and prediction images.

    :param gt_img: The ground truth image.
    :param pred_img: The prediction image.
    """
    # ensure the images are grayscale
    if len(gt_img.shape) > 2:
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    if len(pred_img.shape) > 2:
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)

    # create an empty image for the overlay
    overlay_img = np.zeros((gt_img.shape[0], gt_img.shape[1], 3), dtype=np.uint8)

    # set the ground truth pixels to green
    overlay_img[gt_img > 0] = [0, 255, 0]

    # set the prediction pixels to red
    overlay_img[pred_img > 0] = [0, 0, 255]

    # set the overlapping pixels to white
    overlay_img[(gt_img > 0) & (pred_img > 0)] = [255, 255, 255]

    # create the legend
    cv2.putText(
        overlay_img, "GT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )
    cv2.putText(
        overlay_img, "Pred", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    )

    return overlay_img
