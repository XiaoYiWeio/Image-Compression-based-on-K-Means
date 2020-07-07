# encoding:utf-8

# data_analysis
import numpy as np
import pandas as pd
import math

# Kmeans_model
from sklearn.cluster import KMeans
from sklearn import preprocessing
from skimage import color
from sklearn.cluster import MiniBatchKMeans
from io import BytesIO

# Image_analysis and visualization
import PIL.Image as Image
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload
from mpl_toolkits import mplot3d


# 加载图像，并对数据进行规范化
def load_data(filePath):
    f = open(filePath,'rb')
    # 得到图像的像素值
    img = Image.open(f)
    # 得到图像的尺寸
    width, height = img.size
    data = np.zeros([width, height, 3], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            # 得到点(x,y)的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            data[x][y] = c1, c2, c3
    f.close()
    return data, width, height, img


# 对彩色点像素进行可视化
def plot_pixels(data, title, colors=None, N=10000):
    colors_2 = colors

    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]      # 返回一个list

    colors_1 = data[i]

    colors_2_16 = colors_2[0]
    colors_2_8 = colors_2[1 ]
    colors_2_24 = colors_2[2]

    colors_2_16 = colors_2_16[i]
    colors_2_8 = colors_2_8[i]
    colors_2_24 = colors_2_24[i]

    R, G, B = data[i].T
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    # 绘图， 第二个图用于后面压缩后的图像像素点显示
    ax[0, 0].scatter(R, G, color=colors_1, marker='.')
    ax[0, 0].set(xlabel='RED', ylabel='GREEN', xlim=(0, 1), ylim=(0, 1))

    ax[0, 1].scatter(R, G, color=colors_2_16, marker='.')
    ax[0, 1].set(xlabel='RED', ylabel='GREEN', xlim=(0, 1), ylim=(0, 1))

    ax[1, 0].scatter(R, G, color=colors_2_8, marker='.')
    ax[1, 0].set(xlabel='RED', ylabel='GREEN', xlim=(0, 1), ylim=(0, 1))

    ax[1, 1].scatter(R, G, color=colors_2_24, marker='.')
    ax[1, 1].set(xlabel='RED', ylabel='GREEN', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)
    plt.show()



img, width, height, ori_img = load_data('lena.jpg')
img = img.transpose((1, 0, 2))               # 进行将 0轴 与 1轴调换
img.reshape(width, height, 3)

ax = plt.axes()
ax.imshow(img)
plt.show()

# 三维转二维,用于在色彩空间绘图

data = img/ 255.0
data = data.reshape(width*height, 3)

ori_pixels = data.reshape(*ori_img.size, -1)
'''
# 绘出未压缩前颜色空间
# plot_pixels(data, title='Input color space:255*255*255 ≈ 16,000,000')
X = np.array(ori_img.getdata())
ori_pixels = X.reshape(*ori_img.size, -1)
reload(plt)
fig = plt.figure("3-D Plot of Image")
ax = plt.axes(projection='3d')
for px in X:
    ax.scatter3D(*px, c = np.array([px])/255)
ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")
ax.dist = 11
plt.tight_layout()
plt.show()
'''
# 使用k-means聚类将16million种颜色压缩为16种颜色
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors_0 = kmeans.cluster_centers_[kmeans.predict(data)]
re_colored_0 = new_colors_0.reshape(img.shape)
ax = plt.axes()
ax.imshow(re_colored_0)
plt.show()
'''
reload(plt)
fig = plt.figure("3-D Plot of Image")
ax = plt.axes(projection='3d')
for px in new_colors_0:
    ax.scatter3D(*px, c = np.array([px]))
ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")
ax.dist = 11
plt.tight_layout()
plt.show()
'''
def calculateBCSS(X, kmeans):
    _, label_counts = np.unique(kmeans.labels_, return_counts = True)
    diff_cluster_sq = np.linalg.norm(kmeans.cluster_centers_ - np.mean(X, axis = 0), axis = 1)**2
    return  sum(label_counts * diff_cluster_sq)


def imageByteSize(img):
     img_file = BytesIO()
     image = Image.fromarray(np.uint8(img))
     image.save(img_file, 'png')
     return img_file.tell()/1024

ori_img_n_colors = len(set(ori_img.getdata()))
WCSS = kmeans.inertia_
BCSS = calculateBCSS(data, kmeans)
exp_var = 100*BCSS/(WCSS + BCSS)
print("WCSS: {}".format(WCSS))
print("BCSS: {}".format(BCSS))
print("Explained Variance: {:.3f}%".format(exp_var))
print("Image Size: {:.3f} KB".format(imageByteSize(re_colored_0 * 255.0)))
print()


# 收集多个颜色的数据
range_k_clusters = (2, 24)
ori_img_size = imageByteSize(ori_img)
ori_img_total_variance = sum(np.linalg.norm(data - np.mean(data, axis = 0), axis = 1)**2)
kmeans_result = []
for k in range(*range_k_clusters):
    # CLUSTERING
    kmeans = MiniBatchKMeans(n_clusters=k)
    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
    re_colored = new_colors.reshape(img.shape)

    # EVALUATE
    WCSS = kmeans.inertia_
    BCSS = calculateBCSS(data, kmeans)
    exp_var = 100 * BCSS / (WCSS + BCSS)

    metric = {
        "No. of Colors": k,
        "Pixels": new_colors,
        "WCSS": WCSS,
        "BCSS": BCSS,
        "Explained Variance": exp_var,
        "Image Size (KB)": imageByteSize(new_colors*255.0)
    }

    kmeans_result.append(metric)
kmeans_result = pd.DataFrame(kmeans_result).set_index("No. of Colors")

fig, axes = plt.subplots(3, 3, figsize=(15,15))

# PLOT ORIGINAL IMAGE
axes[0][0].imshow(data.reshape(*ori_img.size, 3))
axes[0][0].set_title("Original Image: {} Colors".format(ori_img_n_colors), fontsize = 20)
axes[0][0].set_xlabel("Image Size: {:.3f} KB".format(ori_img_size), fontsize = 15)
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])

# PLOT COLOR-REDUCED IMAGE
for ax, k, pixels in zip(axes.flat[1:], kmeans_result.index, kmeans_result["Pixels"]):
    compressed_image = np.array(pixels).reshape(*ori_img.size, 3)
    ax.imshow(compressed_image)
    ax.set_title("{} Colors".format(k), fontsize=20)
    ax.set_xlabel("Explained Variance: {:.3f}%\nImage Size: {:.3f} KB".format(kmeans_result.loc[k, "Explained Variance"],
                                                                              kmeans_result.loc[k, "Image Size (KB)"]),
                  fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
fig.suptitle("IMAGE WITH INCREASING NUMBER OF COLORS", size = 30, y = 1.03, fontweight = "bold")
plt.show()

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, metric in zip(axes.flat, kmeans_result.columns[1:]):
    sns.lineplot(x=kmeans_result.index, y=metric, data=kmeans_result, ax=ax)

    if metric == "WCSS":
        y_val = 0
    elif metric == "BCSS":
        y_val = ori_img_total_variance
    elif metric == "Explained Variance":
        y_val = 100
    elif metric == "Image Size (KB)":
        y_val = ori_img_size

    ax.axhline(y=y_val, color='k', linestyle='--', label="Original Image")
    ax.set_xticks(kmeans_result.index[::2])
    ax.ticklabel_format(useOffset=False)
    ax.legend()
plt.tight_layout()
fig.suptitle("METRICS BY NUMBER OF COLORS", size=25, y=1.03, fontweight="bold")
plt.show()


def locateOptimalElbow(x, y):
    # START AND FINAL POINTS
    p1 = (x[0], y[0])
    p2 = (x[-1], y[-1])

    # EQUATION OF LINE: y = mx + c
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = (p2[1] - (m * p2[0]))

    # DISTANCE FROM EACH POINTS TO LINE mx - y + c = 0
    a, b = m, -1
    dist = np.array([abs(a * x0 + b * y0 + c) / math.sqrt(a ** 2 + b ** 2) for x0, y0 in zip(x, y)])
    return np.argmax(dist) + x[0]


def calculateDerivative(data):
    derivative = []
    for i in range(len(data)):
        if i == 0:
            # FORWARD DIFFERENCE
            d = data[i + 1] - data[i]
        elif i == len(data) - 1:
            # BACKWARD DIFFERENCE
            d = data[i] - data[i - 1]
        else:
            # CENTER DIFFERENCE
            d = (data[i + 1] - data[i - 1]) / 2
        derivative.append(d)
    return np.array(derivative)


def locateDrasticChange(x, y):
    # CALCULATE GRADIENT BY FIRST DERIVATIVE
    first_derivative = calculateDerivative(np.array(y))

    # CALCULATE CHANGE OF GRADIENT BY SECOND DERIVATIVE
    second_derivative = calculateDerivative(first_derivative)


    return np.argmax(np.abs(second_derivative)) + x[0]


optimal_k = []
for col in kmeans_result.columns[1:]:
     optimal_k_dict = {}
     optimal_k_dict["Metric"] = col
     if col == "Image Size (KB)":
         optimal_k_dict["Method"] = "Derivative"
         optimal_k_dict["Optimal k"] = locateDrasticChange(kmeans_result.index, kmeans_result[col].values)
     else:
         optimal_k_dict["Method"] = "Elbow"
         optimal_k_dict["Optimal k"] = locateOptimalElbow(kmeans_result.index, kmeans_result[col].values)
     optimal_k.append(optimal_k_dict)
optimal_k = pd.DataFrame(optimal_k)
print(optimal_k)
k_opt = optimal_k["Optimal k"].max()
print("Optimal k : %s" % k_opt)

ori = {
    "Type": "Original",
    "No. of Colors":ori_img_n_colors,
    "Image Size (KB)": ori_img_size,
    "Explained Variance": 100
}
color_reduced = {
    "Type": "Color-Reduced",
    "No. of Colors": k_opt,
    "Image Size (KB)": kmeans_result.loc[k_opt, "Image Size (KB)"],
    "Explained Variance": kmeans_result.loc[k_opt, "Explained Variance"]
}
ori_vs_kmeans = pd.DataFrame([ori, color_reduced]).set_index("Type")
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

print(ori_vs_kmeans)