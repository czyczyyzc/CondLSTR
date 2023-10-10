import cv2
import numpy as np


# x = 1990.992014
# cx = 911.5402
# fy = 1993.371404
# cy = 603.671151
# k1, k2, p1, p2, k3 = -0.551182, 0.289106, 0.000385, -0.001226, 0.0
#
# #相机坐标系到像素坐标系的转换矩阵
# k = np.array([
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0, 0, 1]
# ])
# #畸变系数
# d = np.array([
#     k1, k2, p1, p2, k3
# ])

k = np.array([
    [
        1946.8341507421221,
        0.0,
        940.5830826066215,
    ],
    [
        0.0,
        1944.9608756433358,
        499.88592802367947,
    ],
    [
        0.0,
        0.0,
        1.0
    ],
])


d = np.array(
    [
        -0.5729897286833545,
        0.5293538356713612,
        0.0011837878044178623,
        0.0008613992182658659,
        -0.5949796703249133
    ]
)


def undistort(img):
    h, w = img.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


