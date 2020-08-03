# -*- coding: utf-8 -*-
"""
    Created on Friday, Jul 31 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Saturday, Aug 1 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import cv2
import numpy as np
from math import *


def project(image, position, orientation, fov=120, interval=0.5):
    """
    This function computes the projection of the 2D image from a certain virtual camera
            by giving the position and orientation of the camera.
    :param image: (ndarray) image read by opencv
    :param position: (tuple, 2D coordinate) position of the virtual camera
    :param orientation: (tuple, 2D vector) orientation of the virtual camera
    :param fov: (float, degree) angle range of the virtual camera
    :param interval: (float, degree) angle resolution of the virtual camera
    """
    assert (fov / interval) % 1 == 0
    h, w, c = image.shape

    cvc = np.zeros((h, w, c))
    resolution = int(fov // interval)
    projection = np.zeros((resolution, c))
    projection_slots = [[] for _ in range(resolution)]
    distance_min = np.ones(resolution) * np.inf

    # deg to rad
    fov = radians(fov)
    interval = radians(interval)
    # since the image is in a left-hand coordinate system
    # the angle in the rotate matrix should have an opposite sign
    rotate_mat = np.array([[cos(fov / 2), sin(fov / 2)], [-sin(fov / 2), cos(fov / 2)]])
    orientation = np.array(orientation)
    start_edge = rotate_mat.dot(orientation)  # rotate fov/2 anti-clockwise
    unit_orientation = orientation / np.linalg.norm(orientation)  # unit
    start_edge = start_edge / np.linalg.norm(start_edge)  # unit
    cos_min = unit_orientation.dot(start_edge)

    for y in range(h):
        for x in range(w):
            vector = np.array([x - position[0], y - position[1]])
            pixel_L2 = np.linalg.norm(vector)
            # crop pixel which is out of the view frustum
            cos_angle = vector.dot(unit_orientation) / pixel_L2
            if cos_angle < cos_min:
                continue
            # compute which slot this pixel should be in
            cos_angle = vector.dot(start_edge) / pixel_L2
            angle = acos(cos_angle)
            slot_idx = int(angle / interval)
            projection_slots[slot_idx].append([x, y])
            # compute whether this pixel is the nearest non-zero (surface) in the slot
            if (image[y, x, :] != 0).any() and pixel_L2 < distance_min[slot_idx]:
                distance_min[slot_idx] = pixel_L2
                # update the color value
                projection[slot_idx, :] = image[y, x, :]
    for i, slot in enumerate(projection_slots):
        for pixel in slot:
            cvc[pixel[1], pixel[0], :] = projection[i, :]
    return cvc, projection


def draw_view(image, position, orientation, fov, interval=None):
    fov = radians(fov)
    position = np.array(position)
    rotate_mat1 = np.array([[cos(fov / 2), -sin(fov / 2)], [sin(fov / 2), cos(fov / 2)]])
    rotate_mat2 = np.array([[cos(fov / 2), sin(fov / 2)], [-sin(fov / 2), cos(fov / 2)]])
    edge1 = rotate_mat1.dot(orientation)
    edge2 = rotate_mat2.dot(orientation)
    end1 = (position + 500 * edge1).astype(np.int)
    end2 = (position + 500 * edge2).astype(np.int)
    cv2.line(image, tuple(position), tuple(end1), color=(100, 255, 0), thickness=1)
    cv2.line(image, tuple(position), tuple(end2), color=(100, 255, 0), thickness=1)
    if interval is not None:
        interval = radians(interval)
        assert fov/interval/2 % 1 == 0
        rotate_mat3 = np.array([[cos(interval), -sin(interval)], [sin(interval), cos(interval)]])
        rotate_mat4 = np.array([[cos(interval), sin(interval)], [-sin(interval), cos(interval)]])
        vec1 = orientation
        vec2 = orientation
        mid_end = (position + 500 * orientation).astype(np.int)
        cv2.line(image, tuple(position), tuple(mid_end), color=(100, 255, 0), thickness=1)
        for _ in range(int(fov/interval/2) - 1):
            vec1 = rotate_mat3.dot(vec1)
            vec2 = rotate_mat4.dot(vec2)
            end3 = (position + 500 * vec1).astype(np.int)
            end4 = (position + 500 * vec2).astype(np.int)
            cv2.line(image, tuple(position), tuple(end3), color=(100, 255, 0), thickness=1)
            cv2.line(image, tuple(position), tuple(end4), color=(100, 255, 0), thickness=1)


def draw_cam(image, position, orientation, fov, padding=True, size=100, draw_interval_view=None):
    cam_img = cv2.imread('../../data/cam2.png')
    cam_img = cv2.resize(cam_img, (size, size))
    init_orientation = np.array([0, 1])
    orientation = np.array(orientation)
    orientation = orientation / np.linalg.norm(orientation)
    cosangle = orientation.dot(init_orientation)
    cross_prod = -(init_orientation[0] * orientation[1] - init_orientation[1] * orientation[0])
    direcrion = 1 if cross_prod > 0 else -1
    angle = degrees(acos(cosangle) * direcrion)
    h, w = cam_img.shape[:2]
    M_1 = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    cam_img = cv2.warpAffine(cam_img, M_1, (w, h))
    if padding:
        ih, iw, ic = image.shape
        padding1 = np.zeros((ih, 200, ic))
        padding2 = np.zeros((200, iw + 400, ic))
        padded_img = np.concatenate((padding1, image, padding1), axis=1)
        padded_img = np.concatenate((padding2, padded_img, padding2), axis=0)
    else:
        padded_img = image.copy()
    x = position[0] + 200
    y = position[1] + 200
    padded_img[y - size // 2:y + (size + 1) // 2,
               x - size // 2:x + (size + 1) // 2, :] = cam_img
    draw_view(padded_img, (x, y), orientation, fov, draw_interval_view)
    return padded_img


def black2white(image):
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            if (image[y, x, :] == 0).all():
                image[y, x, :] = np.ones(c) * 255
    return image


if __name__ == '__main__':
    img = cv2.imread('../../data/apple-logo-png-12906.png')

    # set the following parameters, which could change the pose and position of the virtual camera
    position = (-100, 300)
    orientation = (1, 0.3)
    fov = 80

    cvc, projection = project(img, position, orientation, fov, interval=0.5)
    # extend the 1D projection to 64 rows
    projection = projection[np.newaxis, :, :]
    for _ in range(5):
        projection = np.concatenate((projection, projection), axis=0)
    cv2.imwrite('../../images/cvc1_1.png', black2white(cvc))
    cv2.imwrite('../../images/projection1.png', projection)
    layout = draw_cam(img, position, orientation, fov, draw_interval_view=None)

    # set the second camera
    position = (-100, 600)
    orientation = (1, -0.8)
    fov = 80

    cvc, projection = project(img, position, orientation, fov, interval=0.5)
    projection = projection[np.newaxis, :, :]
    for _ in range(5):
        projection = np.concatenate((projection, projection), axis=0)
    cv2.imwrite('../../images/cvc2_2.png', black2white(cvc))
    cv2.imwrite('../../images/projection2.png', projection)
    layout = draw_cam(layout, position, orientation, fov, padding=False)

    layout = layout.astype('uint8')
    black2white(layout)
    # draw layout
    cv2.imwrite('../../images/layout.png', layout)

    # cam = np.zeros((500, 500, 3))
    # cam = draw_cam(cam, (0, 50), (1, 0), 80, padding=False, size=300, draw_interval_view=10)
    # black2white(cam)
    # cv2.imwrite('../../images/vcam.png', cam)

    # cam = np.zeros((500, 500, 3))
    # cam = draw_cam(cam, (0, 50), (1, 0), 80, padding=False, size=300, draw_interval_view=10)
    # black2white(cam)
    # cv2.imwrite('../../images/vcam.png', cam)
