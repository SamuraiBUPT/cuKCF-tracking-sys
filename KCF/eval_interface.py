from kcf_v2 import ObjectTracker as OT2
from kcf_v3 import ObjectTracker as OT3
from kcf_v4 import ObjectTracker as OT4
from kcf_v5 import ObjectTracker as OT5
from kcf_v5_plus import ObjectTracker as OT5_plus
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

import cProfile
import pstats


def calculate_precision(roi_results, gt, thresholds):
    """
    计算精度曲线
    :param roi_results: 预测的目标位置 [(x, y, w, h), ...]
    :param gt: 真实的目标位置 [(x, y, w, h), ...]
    :param thresholds: 距离阈值列表
    :return: 精度曲线
    """
    assert len(roi_results) == len(
        gt), "The number of predictions must match the number of ground truth entries"

    precisions = []
    for t in thresholds:
        correct = 0
        for (px, py, pw, ph), (gx, gy, gw, gh) in zip(roi_results, gt):
            pred_center = np.array([px + pw / 2, py + ph / 2])
            gt_center = np.array([gx + gw / 2, gy + gh / 2])
            distance = np.linalg.norm(pred_center - gt_center)
            if distance <= t:
                correct += 1
        precisions.append(correct / len(gt))
    return precisions


def inference(benchmark_path: str, tracker: str):
    base_path = benchmark_path
    dirs = os.listdir(base_path)

    # iterate each sequence
    all_roi_results = []
    all_gt_results = []
    fps_list = []
    # for seq_path in dirs:
    for seq_path in ['Car1']:
        print(f"evaluating sequence: {seq_path}")
        seq_path = os.path.join(base_path, seq_path)
        imgs_path = os.path.join(seq_path, 'img')
        gt_file = os.path.join(seq_path, 'groundtruth_rect.txt')

        # initialize tracker every sequence
        if tracker == 'kcf_v3':
            tracker_obj = OT3()
        elif tracker == 'kcf_v4':
            tracker_obj = OT4()
        elif tracker == 'kcf_v5':
            tracker_obj = OT5()
        elif tracker == 'kcf_v5_plus':
            tracker_obj = OT5_plus()
        elif tracker == 'kcf_v2':
            tracker_obj = OT2()
        else:
            raise ValueError("Invalid tracker name")

        # read ground truth
        delimiters = [',', '\t', ' ']
        gt = None
        for delimiter in delimiters:
            try:
                data = np.loadtxt(gt_file, delimiter=delimiter)
                gt = data
            except ValueError:
                pass
        # gt = np.loadtxt(gt_file, delimiter=',')
        roi_first = gt[0, :].astype(int)     # first frame roi

        # clear up all the images
        all_images = os.listdir(imgs_path)
        for img_name in all_images:
            appendix = img_name.split('.')[-1]
            if appendix not in ['jpg', 'png', 'jpeg']:
                all_images.remove(img_name)

        # sort the images by name
        all_images.sort()

        # iterate each frame
        cnt = 0
        roi_results = []
        start = time.monotonic()
        for img_name in all_images:
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path)
            if cnt == 0:
                tracker_obj.initialize_first_frame(
                    img, roi_first)
                roi_results.append(roi_first)
            else:
                # print(img)
                if img is None:
                    print(img_path)
                x, y, w, h = tracker_obj.update_tracker(img)
                roi_results.append([x, y, w, h])

            cnt += 1
        end = time.monotonic()
        roi_results = np.array(roi_results)
        # make sure the number of frames is the same
        assert roi_results.shape[0] >= gt.shape[0]
        if roi_results.shape[0] > gt.shape[0]:
            roi_results = roi_results[:gt.shape[0], :]

        # calculate FPS
        fps = cnt / (end - start)
        print(f"FPS: {fps:.2f}")

        fps_list.append(fps)
        all_roi_results.append(roi_results)
        all_gt_results.append(gt)
        # return all_roi_results, all_gt_results, fps_list
    return all_roi_results, all_gt_results, fps_list


def main():
    benchmark_path = './benchmark_dataset'
    results = []
    trackers = [
        # "kcf_v2",
        # "kcf_v3",
        # "kcf_v4",
        # "kcf_v5",
        "kcf_v5_plus"
    ]

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    pec_res = []
    for i, name in enumerate(trackers):

        roi_results, gt_results, fps_list = inference(benchmark_path, name)

        # Flatten the results for all sequences
        roi_results = np.concatenate(roi_results)
        gt = np.concatenate(gt_results)

        thresholds = range(1, 51)  # 1到50的距离阈值
        precision_curve = calculate_precision(roi_results, gt, thresholds)

        avg_fps = np.mean(fps_list)

        print(f"Average FPS: {avg_fps:.2f}")

        # 计算 Mean precision (20 px)
        mean_precision_20px = precision_curve[19]  # 20 px 对应的是索引 19
        print(f"Mean precision (20 px): {mean_precision_20px:.2f}")

        results.append((name, avg_fps, mean_precision_20px))
        pec_res.append(precision_curve)
        color = colors[i % len(colors)]
        plt.plot(thresholds, precision_curve,
                 label=name, color=color)

    # plt.plot(thresholds, precision_curve)
    plt.legend()
    plt.xlabel('Location error threshold')
    plt.ylabel('Precision')
    plt.title('Precision plot for different versions of KCF')
    plt.show()

    with open('tracker_results.txt', 'w') as f:
        for name, avg_fps, mean_precision_20px in results:
            f.write(
                f"{name}: Average FPS: {avg_fps:.2f}, Mean precision (20 px): {mean_precision_20px:.2f}\n")

    # save precision curve results
    np.save('precision_curve_results.npy', np.array(pec_res))


if __name__ == '__main__':
    main()

    # # 可选：性能分析
    # cProfile.run('main()', 'profile_output.prof')

    # # 可选：打印性能分析结果
    # with open('profile_output.txt', 'w') as f:
    #     stats = pstats.Stats('profile_output.prof', stream=f)
    #     stats.strip_dirs().sort_stats('cumulative').print_stats(10)
