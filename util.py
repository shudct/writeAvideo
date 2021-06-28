import cv2
import numpy as np

def convert_video_to_frames(video_path):
    """
    将视频转换成frame array
    :param video_path: 视频路径
    :return: framenum * H * W * 3 array
    """
    video = cv2.VideoCapture(video_path)
    width = int(video.get(3))
    height = int(video.get(4))
    framenum = int(video.get(7))

    frames = np.zeros((framenum, height, width, 3), dtype="uint8")
    cnt = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frames[cnt] = frame
            cv2.imwrite("./frames/"+str(cnt)+".jpg", frame)
            cnt += 1
        else:
            break
    video.release()
    return frames

def compute_hsv_histogram(frame1, frame2):
    """
    计算相邻两帧之间hsv直方图差异
    :param frame1: H * W * 3 array
    :param frame2: H * W * 3 array
    :return:
    """
    frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    hrange = [0, 180]
    srange = [0, 256]
    ranges = hrange + srange
    frame1_hist = cv2.calcHist([frame1_hsv], [0, 1], None, [50, 60], ranges)
    frame2_hist = cv2.calcHist([frame2_hsv], [0, 1], None, [50, 60], ranges)
    frame1_hist_norm = cv2.normalize(frame1_hist, frame1_hist, 0, 255, cv2.NORM_MINMAX)
    frame2_hist_norm = cv2.normalize(frame2_hist, frame2_hist, 0, 255, cv2.NORM_MINMAX)
    return cv2.compareHist(frame1_hist_norm, frame2_hist_norm, 3)   # Bhattacharyya distance

def compute_surf_points(frame1, frame2):
    surf = cv2.ORB_create()
    kp1, des1 = surf.detectAndCompute(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = surf.detectAndCompute(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            goodMatch.append(m)

    return 1-len(goodMatch)/len(kp1)

if __name__ == '__main__':
    # video_path = "/Users/lucydu/Downloads/car.mp4"
    video_path = "./person.mp4"
    frames = convert_video_to_frames(video_path)
    for idx in range(len(frames) - 1):
        hist_inter = compute_hsv_histogram(frames[idx], frames[idx + 1])
        surf_inter = compute_surf_points(frames[idx], frames[idx + 1])
        if hist_inter > 0.2 and surf_inter > 0.8:
            print(idx, hist_inter, surf_inter)

