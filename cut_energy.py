import cv2
import numpy as np
from compute_homography import compute_homography

def cut_avoid_opposite_camera_movements(shot1_path, shot2_path):
    """
    两个镜头连接时，避免相反的运动方向
    :param shot1_path:
    :param shot2_path:
    :return:
    """
    shot1 = cv2.VideoCapture(shot1_path)
    shot2 = cv2.VideoCapture(shot2_path)
    framenum1 = int(shot1.get(7))

    shot1.set(cv2.CAP_PROP_POS_FRAMES, framenum1 - 11)
    ret, shot1_frame1 = shot1.read()
    shot1.set(cv2.CAP_PROP_POS_FRAMES, framenum1 - 1) #最后一帧
    ret, shot1_frame2 = shot1.read()

    shot2.set(cv2.CAP_PROP_POS_FRAMES, 0)             #第一帧
    ret, shot2_frame1 = shot2.read()
    shot2.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, shot2_frame2 = shot2.read()

    H_matrix1 = compute_homography(shot1_frame1, shot1_frame2)
    H_matrix2 = compute_homography(shot2_frame1, shot2_frame2)

    # 前提是shot1和shot2尺寸一致
    W = int(shot1.get(3))
    H = int(shot1.get(4))
    corner0 = np.transpose(np.array([0, 0, 1]))
    corner1 = np.transpose(np.array([0, H, 1]))
    corner2 = np.transpose(np.array([W, 0, 1]))
    corner3 = np.transpose(np.array([W, H, 1]))
    corner = [corner0, corner1, corner2, corner3]

    Q = 0
    for c in corner:
        v_f1 = c - np.dot(H_matrix1, c)
        v_f2 = c - np.dot(H_matrix2, c)
        Q_f1_f2 = np.dot(v_f1, v_f2) / (np.linalg.norm(v_f1, ord=1) * np.linalg.norm(v_f2, ord=1))
        Q += Q_f1_f2

    F_OM = 1 if Q < -0.01 else 0

    return F_OM

def cut_avoid_jump_cuts(shot1_path, shot2_path):
    """

    :param shot1_path:
    :param shot2_path:
    :return:
    """
    shot1 = cv2.VideoCapture(shot1_path)
    shot2 = cv2.VideoCapture(shot2_path)
    framenum1 = int(shot1.get(7))

    shot1.set(cv2.CAP_PROP_POS_FRAMES, framenum1 - 1)  # 最后一帧
    ret, shot1_frame2 = shot1.read()

    shot2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 第一帧
    ret, shot2_frame1 = shot2.read()

    H_matrix = compute_homography(shot1_frame2, shot2_frame1)

    W = int(shot1.get(3))
    H = int(shot1.get(4))
    corner0 = np.transpose(np.array([0, 0, 1]))
    corner1 = np.transpose(np.array([0, H, 1]))
    corner2 = np.transpose(np.array([W, 0, 1]))
    corner3 = np.transpose(np.array([W, H, 1]))
    corner = [corner0, corner1, corner2, corner3]
    sum = 0
    for c in corner:
        sum += np.linalg.norm((np.dot(H_matrix, c) - c))
    score = sum / 4

    F_JC = 1 if score < 150 else 0

    return F_JC


def cut_tonal_continuity(shot1_path, shot2_path):
    """

    :param shot1_path:
    :param shot2_path:
    :return:
    """
    shot1 = cv2.VideoCapture(shot1_path)
    shot2 = cv2.VideoCapture(shot2_path)
    framenum1 = int(shot1.get(7))

    shot1.set(cv2.CAP_PROP_POS_FRAMES, framenum1 - 1)  # 最后一帧
    ret, shot1_frame2 = shot1.read()

    shot2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 第一帧
    ret, shot2_frame1 = shot2.read()

    shot1_frame2_hls = cv2.cvtColor(shot1_frame2, cv2.COLOR_BGR2HLS)
    shot2_frame1_hls = cv2.cvtColor(shot2_frame1, cv2.COLOR_BGR2HLS)

    shot1_hist_l = cv2.calcHist([shot1_frame2_hls], [1], None, [256], [0, 255])
    shot1_hist_s = cv2.calcHist([shot1_frame2_hls], [2], None, [256], [0, 255])
    shot2_hist_l = cv2.calcHist([shot2_frame1_hls], [1], None, [256], [0, 255])
    shot2_hist_s = cv2.calcHist([shot2_frame1_hls], [2], None, [256], [0, 255])
    shot1_hist_l_norm = cv2.normalize(shot1_hist_l, shot1_hist_l, 0, 255, cv2.NORM_MINMAX)
    shot1_hist_s_norm = cv2.normalize(shot1_hist_s, shot1_hist_s, 0, 255, cv2.NORM_MINMAX)
    shot2_hist_l_norm = cv2.normalize(shot2_hist_l, shot2_hist_l, 0, 255, cv2.NORM_MINMAX)
    shot2_hist_s_norm = cv2.normalize(shot2_hist_s, shot2_hist_s, 0, 255, cv2.NORM_MINMAX)

    F_TC = 0.5 * (cv2.compareHist(shot1_hist_l_norm, shot2_hist_l_norm, 3)
                  + cv2.compareHist(shot1_hist_s_norm, shot2_hist_s_norm, 3))

    return F_TC


if __name__ == '__main__':
    shot1_path = "./car_clips/1.mp4"
    shot2_path = "./car_clips/2.mp4"
    F_OM = cut_avoid_opposite_camera_movements(shot1_path, shot2_path)
    print(F_OM)
    F_JC = cut_avoid_jump_cuts(shot1_path, shot2_path)
    print(F_JC)
    F_TC = cut_tonal_continuity(shot1_path, shot2_path)
    print(F_TC)