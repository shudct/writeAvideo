import cv2
import numpy as np
from util import convert_video_to_frames
from compute_final_H import compute_final_H

def shot_saturation_and_brightness(frames):
    """
    一组采样帧的亮度和饱和度的评估值
    :param frames:
    :return:
    """
    F_tone_all = 0
    count = 0
    for idx in range(0, len(frames), 10):
        height, width = frames[idx].shape[0:2]
        frame_hls = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2HLS)
        # 亮度
        frame_lightness = frame_hls[:, :, 1]
        lightness_flag = np.where(frame_lightness > 178.5, 1, 0)

        # 饱和度
        frame_saturation = frame_hls[:, :, 2]
        saturation_flag = np.where(frame_saturation > 178.5, 1, 0)

        score = np.sum(lightness_flag * saturation_flag) / (height * width)
        count += 1
        F_tone_all += score

    return F_tone_all / count

def shot_camera_stability(frames):
    """
    一组采样帧的画面稳定性的评估值
    :param frames:
    :return:
    """

    F_stab_all = 0
    H, W = frames.shape[1:3]
    corner0 = np.transpose(np.array([0, 0, 1]))
    corner1 = np.transpose(np.array([0, H, 1]))
    corner2 = np.transpose(np.array([W, 0, 1]))
    corner3 = np.transpose(np.array([W, H, 1]))
    for idx in range(0, len(frames)-2, 10):
        H12 = compute_final_H(frames[idx], frames[idx + 1])
        H23 = compute_final_H(frames[idx + 1], frames[idx + 2])
        diff0 = np.linalg.norm(np.dot(H23, corner0) - corner0 - (np.dot(H12, corner0) - corner0))
        diff1 = np.linalg.norm(np.dot(H23, corner1) - corner1 - (np.dot(H12, corner1) - corner1))
        diff2 = np.linalg.norm(np.dot(H23, corner2) - corner2 - (np.dot(H12, corner2) - corner2))
        diff3 = np.linalg.norm(np.dot(H23, corner3) - corner3 - (np.dot(H12, corner3) - corner3))
        lcs = (diff0 + diff1 + diff2 + diff3) / 4
        F_stab_all += lcs
    return -F_stab_all/(len(frames)-2)


if __name__ == '__main__':
    video_path = "./car_clips/2.mp4"
    frames = convert_video_to_frames(video_path)
    F_tone = shot_saturation_and_brightness(frames)
    F_stab = shot_camera_stability(frames)
    print(F_tone)
    print(F_stab)
