import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import reshape_as_image
import numpy as np
from matplotlib import pyplot as plt
import random
import cv2 as cv


def segment_images(input_paths, output_path):

    images = [read_image(input_path) for input_path in input_paths]
    segment_mask, edges = segment(images)

    # lets add segmentation mask to first TIF
    dataset = rasterio.open(input_paths[0])
    raster = dataset.read()
    result = np.append(raster, [segment_mask], axis=0)

    profile = dataset.profile
    profile.update(count=4)

    with rasterio.open(output_path, 'w', **profile) as output_dataset:
        output_dataset.write(result)

def segment(images):
    edges = []
    for image in images:
        edge = cv.Canny(image, 50, 100)
        edges.append(edge)

    edges_combined = edges[0]
    for edge in edges[1:]:
        edges_combined = cv.bitwise_or(edges_combined, edge)

    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, threshold = cv.threshold(edges_combined, 0.4, 1.0, cv.THRESH_BINARY)
    # Dilate a bit the dist image
    kernel1 = np.ones((2, 2), dtype=np.uint8)
    dilate = cv.dilate(threshold, kernel1)
    dilate = cv.bitwise_not(dilate) - 254

    dist = cv.distanceTransform(dilate, cv.DIST_L2, 3)
    segments = watershed(dist, images[0])  # TODO: fix passing image here

    return segments, edges_combined


def watershed(distance_image, image):
    dist_8u = distance_image.astype('uint8')
    # Find total markers
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(distance_image.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    cv.watershed(image, markers)

    # Generate random colors
    colors = generate_colors(len(contours))

    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :] = colors[index - 1]

    segments = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    return segments

def generate_colors(n, cmap='gray'):
    colors = []
    cmap = plt.cm.get_cmap(cmap, n)
    for i_label in range(n):
        segment_color = cmap(i_label, alpha=0, bytes=True)
        colors.append((segment_color[0].item(), segment_color[1].item(), segment_color[2].item()))

    random.shuffle(colors)
    return colors


def read_image(image_path):
    raster = rasterio.open(image_path).read()
    image = reshape_as_image(raster)
    return image
