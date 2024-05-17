import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import os

# 提取HOG特征


class HOGDescriptor:
    # 初始化参数
    def __init__(self, detection_window_size):
        self.detection_window_size = detection_window_size
        self.block_size = (8, 8)
        self.block_stride = (4, 4)
        self.cell_size = (4, 4)
        self.num_bins = 9
        self.hog = cv2.HOGDescriptor(detection_window_size, self.block_size, self.block_stride,
                                     self.cell_size, self.num_bins)
    # 计算HOG特征

    def compute_feature(self, image):
        win_stride = self.detection_window_size
        w, h = self.detection_window_size
        w_block, h_block = self.block_stride
        w = w // w_block - 1
        h = h // h_block - 1
        hist = self.hog.compute(
            img=image, winStride=win_stride, padding=(0, 0))
        return hist.reshape(w, h, 36).transpose(2, 1, 0)


class ObjectTracker:
    def __init__(self):
        # 超参数设置
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambda_reg = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.scale_h = 0.
        self.scale_w = 0.
        self.patch_h = 0
        self.patch_w = 0
        self.hog_descriptor = HOGDescriptor((self.patch_w, self.patch_h))
        self.alphaf = None
        self.template = None
        self.roi = None
        self.previous_hist = None

    # 第一帧初始化

    def initialize_first_frame(self, image, roi):
        x, y, w, h = roi
        center_x = x + w // 2
        center_y = y + h // 2
        roi = (center_x, center_y, w, h)

        scale = self.max_patch_size / float(max(w, h))
        self.patch_h = int(h * scale) // 4 * 4 + 4
        self.patch_w = int(w * scale) // 4 * 4 + 4
        self.hog_descriptor = HOGDescriptor((self.patch_w, self.patch_h))
        # print(self.patch_h, self.patch_w)

        feature = self.get_feature(image, roi)
        label = self.gaussian_peak(feature.shape[2], feature.shape[1])

        self.alphaf = self.train(feature, label, self.sigma, self.lambda_reg)
        self.template = feature
        self.roi = roi

        # 提取并存储HSV颜色直方图
        self.previous_hist = self.get_hsv_hist(image, roi)

        # 转换图像为灰度图并存储
        self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 更新tracker位置
    def update_tracker(self, image):
        # 使用HOG特征计算ROI
        hog_roi = self.update_hog_tracker(image)
        # 使用HSV特征计算ROI
        hsv_roi = self.update_hsv_tracker(image)
        # 使用光流特征计算ROI
        flow_roi = self.update_optical_flow(image)

        # 综合决策
        final_roi = self.combine_rois(
            [hog_roi, hsv_roi, flow_roi], [1.5, 0.2, 0.2])

        return final_roi

    # 计算HOG特征，输出ROI
    def update_hog_tracker(self, image):
        cx, cy, w, h = self.roi
        max_response = -1
        best_dx, best_dy = 0, 0
        best_w, best_h = w, h
        best_feature_patch = None

        for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
            roi = map(int, (cx, cy, w * scale, h * scale))
            feature_patch = self.get_feature(image, roi)
            responses = self.detect(self.template, feature_patch, self.sigma)
            height, width = responses.shape
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                best_dx = int((idx % width - width / 2) / self.scale_w)
                best_dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_feature_patch = feature_patch

        # 更新HOG模板
        self.template = self.template * \
            (1 - self.update_rate) + best_feature_patch * self.update_rate
        label = self.gaussian_peak(
            best_feature_patch.shape[2], best_feature_patch.shape[1])
        new_alphaf = self.train(best_feature_patch, label,
                                self.sigma, self.lambda_reg)
        self.alphaf = self.alphaf * \
            (1 - self.update_rate) + new_alphaf * self.update_rate

        # 返回HOG计算的ROI
        return (cx + best_dx, cy + best_dy, best_w, best_h)

    def update_hsv_tracker(self, image):
        cx, cy, w, h = self.roi
        x = int(cx - w // 2)
        y = int(cy - h // 2)
        sub_image = image[y:y + h, x:x + w]
        hsv_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2HSV)
        current_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [
                                    16, 16, 16], [0, 180, 0, 256, 0, 256])
        cv2.normalize(current_hist, current_hist)

        color_similarity = cv2.compareHist(
            self.previous_hist, current_hist, cv2.HISTCMP_CORREL)
        self.previous_hist = current_hist

        best_cx, best_cy = cx, cy
        best_similarity = color_similarity

        for dx in range(-w//2, w//2, 5):
            for dy in range(-h//2, h//2, 5):
                test_roi = (cx + dx, cy + dy, w, h)
                x = int(test_roi[0] - test_roi[2] // 2)
                y = int(test_roi[1] - test_roi[3] // 2)
                sub_image = image[y:y + test_roi[3], x:x + test_roi[2]]
                hsv_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2HSV)
                test_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [
                                         16, 16, 16], [0, 180, 0, 256, 0, 256])
                cv2.normalize(test_hist, test_hist)
                similarity = cv2.compareHist(
                    self.previous_hist, test_hist, cv2.HISTCMP_CORREL)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cx, best_cy = cx + dx, cy + dy

        return (best_cx, best_cy, w, h)

    def update_optical_flow(self, image):
        cx, cy, w, h = self.roi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用Shi-Tomasi角点检测获取前一帧的特征点
        x = int(cx - w // 2)
        y = int(cy - h // 2)
        roi_gray_prev = self.prev_gray[y:y + h, x:x + w]
        p0 = cv2.goodFeaturesToTrack(
            roi_gray_prev, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        if p0 is not None:
            # 转换前一帧特征点坐标为全图坐标
            p0[:, 0, 0] += x
            p0[:, 0, 1] += y

            # 使用Shi-Tomasi角点检测获取当前帧的特征点
            roi_gray_current = gray[y:y + h, x:x + w]
            p1_init = cv2.goodFeaturesToTrack(
                roi_gray_current, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

            if p1_init is not None:
                # 转换当前帧特征点坐标为全图坐标
                p1_init[:, 0, 0] += x
                p1_init[:, 0, 1] += y

                # 计算光流
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, p0, p1_init)

                # 计算有效的运动矢量
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                flow_dx = np.mean(good_new[:, 0] - good_old[:, 0])
                flow_dy = np.mean(good_new[:, 1] - good_old[:, 1])

                # 更新前一帧的灰度图像
                self.prev_gray = gray

                return (cx + int(flow_dx), cy + int(flow_dy), w, h)
            else:
                # 如果当前帧没有检测到特征点，则保持原ROI
                self.prev_gray = gray
                return (cx, cy, w, h)
        else:
            # 如果前一帧没有检测到特征点，则保持原ROI
            self.prev_gray = gray
            return (cx, cy, w, h)

    def combine_rois(self, rois, weights=None):
        if weights is None:
            weights = [1 / len(rois)] * len(rois)

        total_weight = sum(weights)
        final_cx = sum(roi[0] * weight for roi,
                       weight in zip(rois, weights)) / total_weight
        final_cy = sum(roi[1] * weight for roi,
                       weight in zip(rois, weights)) / total_weight
        final_w = sum(roi[2] * weight for roi,
                      weight in zip(rois, weights)) / total_weight
        final_h = sum(roi[3] * weight for roi,
                      weight in zip(rois, weights)) / total_weight

        return int(final_cx - final_w // 2), int(final_cy - final_h // 2), int(final_w), int(final_h)

    # HOG 特征计算
    def get_feature(self, image, roi):
        cx, cy, w, h = roi
        w = int(w * self.padding) // 2 * 2
        h = int(h * self.padding) // 2 * 2
        x = int(cx - w // 2)
        y = int(cy - h // 2)
        # print(x, y, w, h)

        sub_image = image[y:y + h, x:x + w, :]
        resized_image = cv2.resize(
            src=sub_image, dsize=(self.patch_w, self.patch_h))

        if self.gray_feature:
            feature = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(
                1, self.patch_h, self.patch_w) / 255.0 - 0.5
        else:
            feature = self.hog_descriptor.compute_feature(resized_image)

        fc, fh, fw = feature.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w

        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))

        hann2d = hann2t * hann1t
        feature = feature * hann2d
        return feature

    def get_hsv_hist(self, image, roi):
        cx, cy, w, h = roi
        x = int(cx - w // 2)
        y = int(cy - h // 2)
        sub_image = image[y:y + h, x:x + w]
        hsv_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [
                            16, 16, 16], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    # 计算高斯峰值
    def gaussian_peak(self, w, h):
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        half_height, half_width = h // 2, w // 2

        y, x = np.mgrid[-half_height:-half_height +
                        h, -half_width:-half_width + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * \
            np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g
    # 傅立叶变换，计算核相关

    def kernel_correlation(self, x1, x2, sigma):
        fx1 = fft2(x1)
        fx2 = fft2(x2)
        tmp = np.conj(fx1) * fx2
        idft_rbf = ifft2(np.sum(tmp, axis=0))
        idft_rbf = fftshift(idft_rbf)
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * idft_rbf
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def train(self, x, y, sigma, lambdar):
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, x, z, sigma):
        k = self.kernel_correlation(x, z, sigma)
        return np.real(ifft2(self.alphaf * fft2(k)))


def main(video_path):
    video = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(video_name)

    success, image = video.read()
    count = 0

    out_path = 'data/' + video_name
    try:
        os.mkdir(out_path)
    except:
        pass

    while success:
        cv2.imwrite(out_path + '/%08d.jpg' % count, image)
        count += 1
        success, image = video.read()
    video.release()

    base_path = out_path
    tracker = ObjectTracker()
    first_img = cv2.imread(base_path + "/" + "00000000.jpg")
    cv2.imshow("First frame", first_img)
    roi = cv2.selectROI(img=first_img)
    tracker.initialize_first_frame(first_img, roi)
    files = os.listdir(base_path)
    files.sort()
    for file in files:
        if file.endswith('.jpg'):
            path = base_path + "/" + file
            img = cv2.imread(path)
            x, y, w, h = tracker.update_tracker(img)
            print(x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.imshow('Tracking', img)
            c = cv2.waitKey(0)
            if c == 27 or c == ord('q'):
                continue


test_video_path = 'video/Ke.mp4'
main(test_video_path)
