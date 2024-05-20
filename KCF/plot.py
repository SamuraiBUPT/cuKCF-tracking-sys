import numpy as np

import matplotlib.pyplot as plt

# read data from file
data = np.load("precision_curve_results.npy")

data_kcf_v5_plus = [[0.06152032, 0.15887993, 0.24566276, 0.31924365, 0.37893776, 0.42904429, 0.47180794, 0.5113377,  0.54067113, 0.56140618, 0.5809618,  0.59416375, 0.60569168, 0.61413788, 0.62075788, 0.62676914, 0.63266626, 0.63844925, 0.64457465, 0.65035763, 0.65545579, 0.65971694, 0.6633313, 0.66725004,
                     0.67071222, 0.67409831, 0.67870187, 0.68281084, 0.68676762, 0.68969715, 0.69266474, 0.69525186, 0.69761071, 0.69981738, 0.70206209, 0.70457312, 0.70700807, 0.70944301, 0.71130726, 0.71305737, 0.71541622, 0.71750875, 0.71933496, 0.72119921, 0.7231776, 0.72488967, 0.72694415, 0.72873231, 0.73067265, 0.73234667]]

data_kcf_cu = np.array(data_kcf_v5_plus)

result = np.vstack((data, data_kcf_cu))
thresholds = range(1, 51)
color = ['b', 'c', 'y', 'r', 'g', 'm', 'k']
names = ["kcf_v2", "kcf_v3", "kcf_v4", "kcf_v5", "kcf_v5_cuda"]

print(data)

for i in range(result.shape[0]):
    plt.plot(thresholds, result[i], color[i], label=names[i])

plt.legend()
plt.xlabel('Location error threshold')
plt.ylabel('Precision')
plt.title('Precision plot for different versions of KCF')
plt.show()

FPS1 = [27.62, 37.91, 164.85, 89.28]
FPS2 = [27.62, 37.91, 164.85, 89.28, 53.05]
MeanPrecision_20px = [0.18, 0.18, 0.61, 0.64, 0.65]

# 柱状图
labels = ['KCF_v2', 'KCF_v3', 'KCF_v4', 'KCF_v5', 'KCF_v5_cuda']
plt.figure(figsize=(10, 6))
plt.bar(labels, FPS2, color='green')

# 添加标题和标签
plt.title('FPS Comparison of KCF Versions')
plt.xlabel('KCF Versions')
plt.ylabel('FPS')

for i, v in enumerate(FPS2):
    plt.text(i, v + 3, str(v), ha='center', va='bottom')

# 显示图表
plt.show()
