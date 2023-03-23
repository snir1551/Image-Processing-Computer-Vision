import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


# def myID() -> np.int:
#     """
#     Return my ID (not the friend's ID I copied from)
#     :return: int
#     """
#     return


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """

    kernel = np.array([[1, 0, -1]])
    imgX = cv2.filter2D(im2, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    imgY = cv2.filter2D(im2, -1, kernel.T, borderType=cv2.BORDER_REPLICATE)
    imgT = im2 - im1
    size = win_size // 2  # calculate
    originPoints = np.array([])
    newpoints = np.array([])
    start = int(np.floor(win_size / 2))
    # get the the of the images
    for i in range(start, im1.shape[0] - start, step_size):
        for j in range(start, im1.shape[1] - start, step_size):

            AA = np.array([[(imgX[i - size: i + size + 1, j - size: j + size + 1] * imgX[i - size: i + size + 1,
                                                                          j - size: j + size + 1]).sum(), (
                                        imgX[i - size: i + size + 1, j - size: j + size + 1] * imgY[i - size: i + size + 1,
                                                                                     j - size: j + size + 1]).sum()],
                           [(imgX[i - size: i + size + 1, j - size: j + size + 1] * imgY[i - size: i + size + 1,
                                                                          j - size: j + size + 1]).sum(), (
                                    imgY[i - size: i + size + 1, j - size: j + size + 1] * imgY[i - size: i + size + 1,
                                                                                 j - size: j + size + 1]).sum()]])
            BB = np.array(
                [(imgX[i - size: i + size + 1, j - size: j + size + 1] * imgT[i - size: i + size + 1, j - size: j + size + 1]).sum(), (
                        imgY[i - size: i + size + 1, j - size: j + size + 1] * imgT[i - size: i + size + 1, j - size: j + size + 1]).sum()])

            t, t1 = np.linalg.eig(AA)
            t = np.sort(t)

            if t[0] > 1 and t[1] / t[0] < 100:  # conditions
                vec = np.linalg.inv(AA) @ (BB)

                n = [vec[0], vec[1]]
                o = [j, i]

                originPoints = np.append(originPoints, o)  # adding the points
                newpoints = np.append(newpoints, n)

    originPoints = originPoints.reshape(int(originPoints.shape[0] / 2), 2)  # change the shape to pair
    newpoints = newpoints.reshape(int(newpoints.shape[0] / 2), 2)

    return originPoints, newpoints


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    # make the pyramid of image1 and image2 (x levels)
    matA = []  # declare
    matB = []  # declare
    matA.append(img1)
    matB.append(img2)

    for i in range(k - 1):
        matA.append(cv2.pyrDown(matA[-1], (matA[-1].shape[0] // 2, matA[-1].shape[1] // 2)))
        matB.append(cv2.pyrDown(matB[-1], (matB[-1].shape[0] // 2, matB[-1].shape[1] // 2)))

    matrixC = []  # make the pyramid of the change
    for i in range(len(matA)):
        a = matA[i].shape[0]  # shape
        b = matA[i].shape[1]  # shape
        change = np.zeros((a, b, 2))
        old, new = opticalFlow(matA[i], matB[i], step_size=stepSize, win_size=winSize)
        for x in range(len(old)):
            b = old[x][0].astype(int)
            a = old[x][1].astype(int)
            c = new[x][0]
            d = new[x][1]
            change[a][b][0] = c
            change[a][b][1] = d
        matrixC.append(change)

    for x in range(-1, -k, -1):  # run on the range between -1 k -1 on the matrix
        y = x - 1
        for i in range(matrixC[x].shape[0]):
            for j in range(matrixC[x].shape[1]):
                matrixC[y][i * 2][j * 2][0] += (matrixC[x][i][j][0] * 2)  # calculate
                matrixC[y][i * 2][j * 2][1] += (matrixC[x][i][j][1] * 2)  # calculate

    return matrixC[0]


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    diff = float("inf")
    spot = 0
    old, new = opticalFlow(im1, im2, 10, 5)

    for x in range(len(new)):  # see all the v,u we found
        t1 = new[x][0]
        t2 = new[x][1]
        t = np.array([[1, 0, t1],
                      [0, 1, t2],
                      [0, 0, 1]], dtype=np.float)

        newimg = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))  # make new img transformation using u,v

        d = ((im2 - newimg) ** 2).sum()  # find difference in img

        if d < diff:
            diff = d
            spot = x
            if diff == 0:
                print("break")
                break
    t1 = new[spot][0]
    t2 = new[spot][1]

    t = np.array([[1, 0, t1],
                  [0, 1, t2],
                  [0, 0, 1]], dtype=np.float)

    return t


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    old, new = opticalFlow(im1, im2, 10, 5)
    diff = float("inf")
    spot = 0

    for n in range(new.shape[0]):  # see all the v,u
        x = new[n][0]
        y = new[n][1]

        if x != 0:
            theta = np.arctan(y / x)
        else:
            theta = 0

        trans = np.array([[np.cos(theta), -np.sin(theta), 0],  # make new img transformation using theta
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]], dtype=np.float)

        newpic = cv2.warpPerspective(im1, trans, (im1.shape[1], im1.shape[0]))

        d = ((im2 - newpic) ** 2).sum()  # finding difference in image
        if d < diff:
            diff = d
            spot = n
        if diff == 0:
            break

    # find the uv
    x = new[spot][0]
    y = new[spot][1]
    if x != 0:
        theta = np.arctan(y / x)
    else:
        theta = 0
    t = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]], dtype=np.float)
    newpic = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

    mat = findTranslationLK(newpic, im2)
    T = mat @ t
    return (T)


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    win = 13
    pad = win // 2
    im2pad = cv2.copyMakeBorder(im2, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)

    # middle of the window
    I = []
    J = []
    for x in range(1, 5):
        I.append((im1.shape[0] // 5) * x)
        J.append((im1.shape[1] // 5) * x)

    corr_listt = [(np.array([0]), 0, 0)]
    for x in range(len(I)):
        for y in range(len(J)):
            windowa = im1[I[x] - pad:I[x] + pad + 1, J[y] - pad:J[y] + pad + 1]  # template to match
            a = windowa.reshape(1, win * win)
            aT = a.T
            big = [(np.array([0]), 0, 0)]
            for i in range(0, im2.shape[0]):
                for j in range(0, im2.shape[1]):
                    if (i + pad + win) < im2pad.shape[0] and (j + pad + win) < im2pad.shape[1]:
                        windowb = im2pad[i + pad:i + pad + win, j + pad:j + pad + win]
                        b = windowb.reshape(1, win * win)
                        bT = b.T
                        top = np.dot(a, bT)
                        bottom = np.dot(a, aT) + np.dot(b, bT)
                        if bottom != 0:  # finding the correlation between this window and template
                            corr = top / bottom
                            if corr > big[0][0]:
                                big.clear()
                                big.insert(0, (corr, i, j))
                            elif corr == big[0][0]:
                                big.insert(0, (corr, i, j))
            if big[0][0][0] > corr_listt[0][0][
                0]:
                corr_listt.clear()
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))
            if big[0][0][0] == corr_listt[0][0][
                0]:  # copy the values from big and add the x y values of the original image)
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))

    dif = float("inf")
    spot = -1
    for x in range(len(corr_listt)):

        t1 = corr_listt[x][1][0] - corr_listt[x][0][1]  # u
        t2 = corr_listt[x][1][1] - corr_listt[x][0][2]  # v
        t = np.array([[1, 0, t1],  # create new image with the found transformation
                      [0, 1, t2],
                      [0, 0, 1]], dtype=np.float)
        new = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))
        d = ((im2 - new) ** 2).sum()
        if d < dif:
            dif = d
            spot = x
            if dif == 0:
                break

    t1 = corr_listt[spot][1][0] - corr_listt[spot][0][1]  # u ,look and take the smallest diff and return transformation
    t2 = corr_listt[spot][1][1] - corr_listt[spot][0][2]  # v ,look and take the smallest diff and return transformation
    t = np.array([[1, 0, t1],
                  [0, 1, t2],
                  [0, 0, 1]], dtype=np.float)
    return t


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    win = 13
    pad = win // 2

    im2pad = cv2.copyMakeBorder(im2, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)


    I = []
    J = []
    for x in range(1, 5):
        I.append((im1.shape[0] // 5) * x)
        J.append((im1.shape[1] // 5) * x)

    corr_listt = [(np.array([0]), 0, 0)]
    for x in range(len(I)):
        for y in range(len(J)):

            windowa = im1[I[x] - pad:I[x] + pad + 1, J[y] - pad:J[y] + pad + 1]
            a = windowa.reshape(1, win * win)
            aT = a.T
            big = [(np.array([0]), 0, 0)]
            for i in range(0, im2.shape[0]):
                for j in range(0, im2.shape[1]):
                    if (i + pad + win) < im2pad.shape[0] and (j + pad + win) < im2pad.shape[1]:
                        windowb = im2pad[i + pad:i + pad + win, j + pad:j + pad + win]
                        b = windowb.reshape(1, win * win)
                        bT = b.T
                        top = np.dot(a, bT)
                        bottom = np.dot(a, aT) + np.dot(b, bT)

                        if bottom != 0:
                            corr = top / bottom
                            if corr > big[0][0]:
                                big.clear()
                                big.insert(0, (corr, i, j))
                            elif corr == big[0][0]:
                                big.insert(0, (corr, i, j))

            if big[0][0][0] > corr_listt[0][0][
                0]:
                corr_listt.clear()
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))
            if big[0][0][0] == corr_listt[0][0][
                0]:  # (copy the values from big and add the x y values of the original image)
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))

    spot = -1
    diff = float("inf")

    for n in range(len(corr_listt)):
        x = corr_listt[n][1][0] - corr_listt[n][0][1]
        y = corr_listt[n][1][1] - corr_listt[n][0][2]

        if (x != 0):
            theta = np.arctan(y / x)
        else:
            theta = 0

        t = np.array([[np.cos(theta), -np.sin(theta), x],
                      [np.sin(theta), np.cos(theta), y],
                      [0, 0, 1]], dtype=np.float)

        newimg = cv2.warpPerspective(im1, t, im1.shape[::-1])

        d = ((im2 - newimg) ** 2).sum()
        if d < diff:
            diff = d
            spot = n
        if diff == 0:
            break

    x = corr_listt[spot][1][0] - corr_listt[spot][0][1]
    y = corr_listt[spot][1][1] - corr_listt[spot][0][2]

    if x != 0:
        theta = np.arctan(y / x)
    else:
        theta = 0
    t = np.array([[np.cos(theta), -np.sin(theta), x],
                  [np.sin(theta), np.cos(theta), y],
                  [0, 0, 1]], dtype=np.float)

    return t


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    new = np.zeros((im1.shape[0], im1.shape[1]))
    Tinv = np.linalg.inv(T)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            arr = np.array([i, j, 1])
            newarr = Tinv @ arr
            x1 = np.floor(newarr[0]).astype(int)
            x2 = np.ceil(newarr[0]).astype(int)
            x3 = round(newarr[0] % 1, 3)
            y1 = np.floor(newarr[1]).astype(int)
            y2 = np.ceil(newarr[1]).astype(int)
            y3 = round(newarr[1] % 1, 3)

            if x1 >= 0 and y1 >= 0 and x1 < im1.shape[0] and y1 < im1.shape[1]:
                new[i][j] += (1 - x3) * (1 - y3) * im1[x1][y1]

            if x2 >= 0 and y1 >= 0 and x2 < im1.shape[0] and y1 < im1.shape[1]:
                new[i][j] += x3 * (1 - y3) * im1[x2][y1]

            if x1 >= 0 and y2 >= 0 and x1 < im1.shape[0] and y2 < im1.shape[1]:
                new[i][j] += (1 - x3) * y3 * im1[x1][y2]

            if x2 >= 0 and y2 >= 0 and x2 < im1.shape[0] and y2 < im1.shape[1]:
                new[i][j] += x3 * y3 * im1[x2][y2]

    plt.imshow(new)
    plt.show()

    return new


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


# used last task
def blur_Image_2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    assert (kernel_size % 2 == 1)

    sigma = int(round(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8))
    kernel = cv2.getGaussianKernel(kernel_size, sigma)

    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    img = img[0: np.power(2, levels) * int(img.shape[0] / np.power(2, levels)),
          0: np.power(2, levels) * int(img.shape[1] / np.power(2, levels))]

    temp_img = img.copy()
    pyr = [temp_img]
    for i in range(levels - 1):
        temp_img = blur_Image_2(temp_img, 5)
        temp_img = temp_img[::2, ::2]
        pyr.append(temp_img)

    return pyr


# used last task
def kernel_gaussian(kernel_size: int):
    """
    kernelGaussian
    :param kernel_size: size of the kernel
    :return:
    """

    sigma = int(round(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8))
    g_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    g_kernel = g_kernel * g_kernel.transpose()

    return g_kernel



def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    expand = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    expand[::2, ::2] = img
    expand = cv2.filter2D(expand, -1, gs_k, borderType=cv2.BORDER_REPLICATE)

    return expand


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyramid = []
    g_ker = kernel_gaussian(5)
    g_ker *= 4
    gaussian_pyr = gaussianPyr(img, levels)
    for i in range(levels - 1):
        extend_level = gaussExpand(gaussian_pyr[i + 1], g_ker)
        lap_level = gaussian_pyr[i] - extend_level
        pyramid.append(lap_level.copy())
    pyramid.append(gaussian_pyr[-1])

    return pyramid


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pyr_updated = lap_pyr.copy()
    guss_k = kernel_gaussian(5) * 4
    cur_layer = lap_pyr[-1]
    for i in range(len(pyr_updated) - 2, -1, -1):
        cur_layer = gaussExpand(cur_layer, guss_k) + pyr_updated[i]

    return cur_layer


# for pyrBlend
def pyrBlend_function(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """
        Blends two images using PyramidBlend method
        :param img_1: Image 1
        :param img_2: Image 2
        :param mask: Blend mask
        :param levels: Pyramid depth
        :return:  Blended Image
        """
    L1 = laplaceianReduce(img_1, levels)
    L2 = laplaceianReduce(img_2, levels)
    Gm = gaussianPyr(mask, levels)
    Lout = []
    for k in range(levels):
        curr_lup = Gm[k] * L1[k] + (1 - Gm[k]) * L2[k]
        Lout.append(curr_lup)
    imageBlend = laplaceianExpand(Lout)
    imageBlend = np.clip(imageBlend, 0, 1)

    return imageBlend


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    assert (img_1.shape == img_2.shape)  # check if this is the same img shape

    img_1 = img_1[0: np.power(2, levels) * int(img_1.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img_1.shape[1] / np.power(2, levels))]
    img_2 = img_2[0: np.power(2, levels) * int(img_2.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img_2.shape[1] / np.power(2, levels))]
    mask = mask[0: np.power(2, levels) * int(mask.shape[0] / np.power(2, levels)),
           0: np.power(2, levels) * int(mask.shape[1] / np.power(2, levels))]

    im_blend = np.zeros(img_1.shape)
    if len(img_1.shape) == 3 or len(img_2.shape) == 3:  # check if the image is RGB
        for color in range(3):
            part_im1 = img_1[:, :, color]
            part_im2 = img_2[:, :, color]
            part_mask = mask[:, :, color]
            im_blend[:, :, color] = pyrBlend_function(part_im1, part_im2, part_mask, levels)

    else:
        im_blend = pyrBlend_function(img_1, img_2, mask, levels)  # check if the image is grayscale

    naive_blend = mask * img_1 + (1 - mask) * img_2
    return naive_blend, im_blend

