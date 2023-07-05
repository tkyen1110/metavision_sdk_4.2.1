# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Script to visualize extrinsic camera parameters.
"""

import cv2 as cv
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os


def parse_args():
    import argparse
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize extrinsic camera parameters.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i,--input-file', dest='input_file', type=str, default='/tmp/mono_calibration/intrinsics.json',
        help='Json camera calibration file containing the pattern geometry (n_cols, n_rows, square_height and '
        'square_width) and the extrinsics (rvecs and tvecs, matrices of shape (n_views,3), or views, a list of '
        'n_views matrices '
        'of shape (4,4)). ')
    parser.add_argument(
        '-m', '--matrix-format', dest='matrix_format', default=False, action='store_true',
        help='If specified, the app assumes 4x4 pose matrices and will search for them in \'views\' under the '
        '\'T_c_w\' key. By default, the app assumes Rodrigues rotation vectors and translation vectors, as '
        'provided by OpenCV, and will search for them under the \'rvecs\' and \'tvecs\' keys.')
    parser.add_argument(
        '-p', '--pattern-centric', dest='pattern_centric', default=False, action='store_true',
        help='If specified, the app will display the camera locations in the pattern coordinate system. By '
        'default, the app displays the pattern locations in the camera coordinate system.'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print("The input extrinsics file does not exist.")
        print(args.input_file)
        exit(1)

    return args


def draw_extrinsics(ax, pattern_width, pattern_height, T_cp_poses, pattern_centric=False):
    """
    Visualizes camera extrinsics from 4x4 pose matrices. By default display pattern locations in camera frame.

    Args:
        ax (axes.SubplotBase): Axes of the 3D plot
        pattern_width (float): Pattern width (in m)
        pattern_height (float): Pattern height (in m)
        T_cp_poses (list of np.array of shape (4x4)): List of the pattern poses in the camera frame (extrinsics)
        pattern_centric (bool): If True, will display camera locations in pattern frame
    """

    # Polygon in the pattern frame (Each column is a homogeneous 3d point)
    Poly_p = np.zeros((4, 5))
    Poly_p[:, 0] = [0, 0, 0, 1]
    Poly_p[:, 1] = [pattern_width, 0, 0, 1]
    Poly_p[:, 2] = [pattern_width, pattern_height, 0, 1]
    Poly_p[:, 3] = [0, pattern_height, 0, 1]
    Poly_p[:, 4] = [0, 0, 0, 1]

    # Draw the camera plane (Each column is a homogeneous 3d point)
    Cam_c = np.zeros((4, 5))
    Cam_c[:, 0] = [0, 0, 0, 1]
    Cam_c[:, 1] = [0.05, 0, 0, 1]
    Cam_c[:, 2] = [0.05, 0.05, 0, 1]
    Cam_c[:, 3] = [0, 0.05, 0, 1]
    Cam_c[:, 4] = [0, 0, 0, 1]

    n_views = len(T_cp_poses)
    cm_subsection = np.linspace(0.0, 1.0, n_views)
    colors = [cm.jet(x) for x in cm_subsection]

    # Pose mapping from camera to matplotlib frame
    T_mc = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]])

    if not pattern_centric:
        # draw camera as origin
        Cam_m = T_mc.dot(Cam_c)
        ax.plot3D(Cam_m[0, :], Cam_m[1, :], Cam_m[2, :], color="green")
        ax.scatter(Cam_m[0, 0], Cam_m[1, 0], Cam_m[2, 0], color="green")
        ax.text(Cam_m[0, 0], Cam_m[1, 0], Cam_m[2, 0], 'C', size=10, zorder=1, color='k')

        min_values = np.minimum(np.zeros(3), Cam_m[0:3, :].min(axis=1))
        max_values = np.maximum(np.zeros(3), Cam_m[0:3, :].max(axis=1))

        # draw each pattern position
        for i, (T_cp, color) in enumerate(zip(T_cp_poses, colors)):
            # Polygon in the matplotlib frame
            Poly_m = T_mc.dot(T_cp.dot(Poly_p))

            ax.plot3D(Poly_m[0, :], Poly_m[1, :], Poly_m[2, :], color=color)
            ax.scatter(Poly_m[0, 0], Poly_m[1, 0], Poly_m[2, 0], color=color)
            ax.text(Poly_m[0, 0], Poly_m[1, 0], Poly_m[2, 0],  '%s' % (str(i+1)), size=10, zorder=1, color='k')
            min_values = np.minimum(min_values, Poly_m[0:3, :].min(axis=1))
            max_values = np.maximum(max_values, Poly_m[0:3, :].max(axis=1))

        ax.set_title('Extrinsic Camera Parameters Visualization, camera coordinate system')

    else:
        # draw pattern as origin
        Poly_m = T_mc.dot(Poly_p)
        ax.plot3D(Poly_m[0, :], Poly_m[1, :], Poly_m[2, :], color="green")
        ax.scatter(Poly_m[0, 0], Poly_m[1, 0], Poly_m[2, 0], color="green")
        ax.text(Poly_m[0, 0], Poly_m[1, 0], Poly_m[2, 0],  'P', size=10, zorder=1, color='k')

        min_values = np.minimum(np.zeros(3), Poly_m[0:3, :].min(axis=1))
        max_values = np.maximum(np.zeros(3), Poly_m[0:3, :].max(axis=1))

        # draw each camera position
        for i, (T_cp, color) in enumerate(zip(T_cp_poses, colors)):
            # Polygon in the matplotlib frame
            T = np.linalg.inv(T_cp)
            Cam_m = T_mc.dot(T.dot(Cam_c))

            ax.plot3D(Cam_m[0, :], Cam_m[1, :], Cam_m[2, :], color=color)
            ax.scatter(Cam_m[0, 0], Cam_m[1, 0], Cam_m[2, 0], color=color)
            ax.text(Cam_m[0, 0], Cam_m[1, 0], Cam_m[2, 0],  '%s' % (str(i+1)), size=10, zorder=1, color='k')
            min_values = np.minimum(min_values, Cam_m[0:3, :].min(axis=1))
            max_values = np.maximum(max_values, Cam_m[0:3, :].max(axis=1))

        ax.set_title('Extrinsic Camera Parameters Visualization, pattern coordinate system')

    # Set 3D bounds
    half_max_range = 0.5 * (max_values - min_values).max()
    mid = 0.5*(max_values + min_values)
    ax.set_xlim(mid[0]-half_max_range, mid[0]+half_max_range)
    ax.set_ylim(mid[1]-half_max_range, mid[1]+half_max_range)
    ax.set_zlim(mid[2]-half_max_range, mid[2]+half_max_range)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')


def draw_extrinsics_rodrigues(ax, pattern_width, pattern_height, rvecs, tvecs, pattern_centric=False):
    """
    Visualizes camera extrinsics from translations and Rodrigues' rotations.

    Args:
        ax (axes.SubplotBase): Axes of the 3D plot
        pattern_width (float): Pattern width (in m)
        pattern_height (float): Pattern height (in m)
        rvecs (np.array of shape (nx3)): Rodrigues' rotation vectors of the patterns in the camera frame
        tvecs (np.array of shape (nx3)): Translation vectors of the patterns in the camera frame
        pattern_centric (bool): If True, will display camera locations in pattern frame
    """

    T_cp_poses = []
    n_views = rvecs.shape[0]
    for idx in range(n_views):
        # Pose mapping from pattern to camera frame
        R, _ = cv.Rodrigues(rvecs[idx])
        T_cp = np.zeros((4, 4))
        T_cp[0:3, 0:3] = R
        T_cp[0:3, 3] = tvecs[idx]
        T_cp[3, :] = [0, 0, 0, 1]
        T_cp_poses.append(T_cp)

    draw_extrinsics(ax, pattern_width, pattern_height, T_cp_poses, pattern_centric)


def main():
    """ Main """
    args = parse_args()

    print("Tool for visualizing extrinsic camera parameters.")

    fig = plt.figure('Extrinsic Camera Parameters Visualization')
    ax = fig.add_subplot(111, projection='3d')

    with open(args.input_file, 'r') as f:
        fs_dico = json.load(f)
        fs_pattern = fs_dico["pattern"]
        n_cols = fs_pattern["n_cols"]
        n_rows = fs_pattern["n_rows"]
        pattern_width = (n_cols-1) * fs_pattern["square_width"]
        pattern_height = (n_rows-1) * fs_pattern["square_height"]

        if args.matrix_format:
            T_cp_poses = [np.array(v["T_c_w"]) for v in fs_dico["views"]]
            draw_extrinsics(ax, pattern_width, pattern_height, T_cp_poses, args.pattern_centric)
        else:
            fs = cv.FileStorage(args.input_file, cv.FILE_STORAGE_READ)
            rvecs = fs.getNode('rvecs').mat()
            tvecs = fs.getNode('tvecs').mat()
            if rvecs.shape[1] != 3 or tvecs.shape[1] != 3:
                nelem = rvecs.size
                rvecs = np.reshape(rvecs, (nelem//3, 3))
                tvecs = np.reshape(tvecs, (nelem//3, 3))
            draw_extrinsics_rodrigues(ax, pattern_width, pattern_height, rvecs, tvecs, args.pattern_centric)
            fs.release()

    plt.show()


if __name__ == '__main__':
    main()
