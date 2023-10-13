import numpy as np
import scipy.spatial.distance as dist


def crop_det(img, rect):
    tl, br = rect
    return img[tl[1]:br[1] + 1, tl[0]:br[0] + 1]

    
def order_points(pts):
    pts = np.array(pts)
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="int32")


def box2rec(box, shape):
    tl, tr, br, bl = order_points(box)
    tl[np.where(tl < 0)] = 0
    for i in range(2):
        br[np.where(br >= shape[i])] = shape[i] - 1

    return list(tl), list(br)


def is_valid_rec(tl, br):
    return tl[0] < br[0] and tl[1] < br[1]


def rect2box(rect):
    try:
        iterator = iter(rect)
    except:
        return None
    
    tl, br = rect

    return [tl, [tl[0], br[1]], br, [br[0], tl[1]]]


def rbox2bbox(rbox):
    if isinstance(rbox, float):
        return np.nan, np.nan, np.nan, np.nan

    xs, ys = zip(*eval(rbox))

    return min(xs), min(ys), max(xs), max(ys)