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
        # print(feature.shape)
        label = self.gaussian_peak(feature.shape[2], feature.shape[1])

        self.alphaf = self.train(feature, label, self.sigma, self.lambda_reg)
        self.template = feature
        self.roi = roi
    # 更新tracker位置

    def update_tracker(self, image):
        cx, cy, w, h = self.roi
        max_response = -1
        # print(cx, cy, w, h)
        for scale in [0.8, 1.0, 1.2]:
            roi = map(int, (cx, cy, w * scale, h * scale))
            feature_patch = self.get_feature(image, roi)
            responses = self.detect(self.template, feature_patch, self.sigma)
            height, width = responses.shape
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_feature_patch = feature_patch
        # 更新roi位置
        self.roi = (cx + dx, cy + dy, best_w, best_h)
        # 更新template
        self.template = self.template * \
            (1 - self.update_rate) + best_feature_patch * self.update_rate
        label = self.gaussian_peak(
            best_feature_patch.shape[2], best_feature_patch.shape[1])
        new_alphaf = self.train(best_feature_patch, label,
                                self.sigma, self.lambda_reg)
        self.alphaf = self.alphaf * \
            (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return cx - w // 2, cy - h // 2, w, h

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


def inference(video_path):
    main(video_path)


if __name__ == '__main__':
    test_video_path = 'video/car.mp4'
    main(test_video_path)
