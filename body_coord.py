#!/usr/bin/env python
import cv2
import numpy as np

"""
coordinates (x,y,z)
x - column direction of image plane
y - row direction of image plane
z - depth
"""

def getCentroid(img):
    (row_list, col_list) = np.where(img == 0)
    centroid_row = np.sum(row_list)/(row_list.size)
    centroid_col = np.sum(col_list)/(col_list.size)
    return centroid_col, centroid_row

def body_coord(depth_image):
    # threshold the depth image to get human shape
    retval2, threshold = cv2.threshold(depth_image, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find centroid of detected human
    threshold = np.array(threshold)
    body_coord_col, body_coord_row = getCentroid(threshold)
    return (body_coord_col, body_coord_row)

def position_3D(point, depth, frame):
    (col, row) = np.subtract(point, frame)
    return (col, row, depth)

def plot_vector(img, point1, point2):
    # plot a vector from point2 to point1
    cv2.circle(img, point1, 5, (0,255,0), 4)
    cv2.line(img, point1, point2, (0,255,0),2)
    return img