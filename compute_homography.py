import cv2
import numpy as np

def compute_homography(frame1, frame2):
    """

    :param frame1:
    :param frame2:
    :return:
    """
    orb = cv2.ORB_create()
    kpt1, des1 = orb.detectAndCompute(frame1, None)
    kpt2, des2 = orb.detectAndCompute(frame2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M