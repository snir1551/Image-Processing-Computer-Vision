"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import matplotlib.pyplot as plt
import cv2
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


# def myID() -> np.int:
#     """
#     Return my ID (not the friend's ID I copied from)
#     :return: int
#     """
#     return


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:  # GRAYSCALE
        result_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # RGB
        result_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(gray_scale_img.shape)
    norm_img = result_img / 255.0  # normalize

    return norm_img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)  # get conversation img
    if representation == LOAD_GRAY_SCALE:  # GRAYSCALE
        cv2.imshow("GrayScaleImage", img)  # load img
        cv2.waitKey(0)  # show img in format cv2
    else:  # RGB
        plt.imshow(img)  # load img
        plt.show()  # show img in matplotlib format


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    matrix_rgb = matrix()  # get matrix

    yiq = np.dot(imgRGB, matrix_rgb.T.copy())  # multiplication
    return yiq


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    matrix_rgb = matrix()  # get matrix
    matrix_yiq = np.linalg.inv(matrix_rgb)  # reverse matrix
    rgb = np.dot(imgYIQ, matrix_yiq.T.copy())  # multiplication
    return rgb


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    flagRGB = False

    if len(imgOrig.shape) == 3:  # if RGB image
        flagRGB = True
        yiqIm = transformRGB2YIQ(imgOrig)  # Convert from RGB to YIQ
        imgOrig = yiqIm[:, :, 0]  # take the Y channel

    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)  # normalize
    imgOrig = imgOrig.astype('uint8')  # make sure all pixels are integers

    histOrg = np.histogram(imgOrig.flatten(), bins=256)[0]  # take the hist from the histogram function
    cs = np.cumsum(histOrg)  # calculate cumsum on the hist

    imgEQ = cs[imgOrig]  # the new img
    imgEQ = cv2.normalize(imgEQ, None, 0, 255, cv2.NORM_MINMAX)  # normalize
    imgEQ = imgEQ.astype('uint8')  # make sure all pixels are integers

    histEQ = np.histogram(imgEQ.flatten(), bins=256)[0]  # get the hist of the new img

    if flagRGB:
        yiqIm[:, :, 0] = imgEQ / 255  # normalize
        imgEQ = transformYIQ2RGB(yiqIm)  # Convert from YIQ to RGB

    return imgEQ, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    flagRGB = bool(imOrig.ndim == 3)  # RGB image
    if flagRGB:  # RGB
        imgYIQ = transformRGB2YIQ(imOrig)  # Convert to YIQ
        imOrig = np.copy(imgYIQ[:, :, 0])  # save Y channel
    else:  # GRAYSCALE
        imgYIQ = imOrig

    if np.amax(imOrig) <= 1:  # its means that imOrig is normalized
        imOrig = imOrig * 255
    imOrig = imOrig.astype('uint8')  # make sure all pixels are integers

    histORGN = np.histogram(imOrig.flatten(), bins=256)[0]  # Calculate a histogram of the original image

    # find the boundaries
    size = int(255 / nQuant)  # Divide the intervals evenly
    _Z = np.zeros(nQuant + 1, dtype=int)  # _Z is an array that will represents the boundaries
    for i in range(1, nQuant):  # move on the colors to quantize
        _Z[i] = _Z[i - 1] + size  # boundary coordinates
    _Z[nQuant] = 255  # The left border will always start at 0 and the right border will always end at 255
    _Q = np.zeros(nQuant)  # _Q is an array that represent the values of the boundaries

    quantized_lst = list()
    MSE_lst = list()

    for i in range(nIter):  # loop nInter times
        _newImg = np.zeros(imOrig.shape)  # Initialize a matrix with 0 in the original image size

        for j in range(len(_Q)):  # every j is a cell
            if j == len(_Q) - 1:  # last itarate of j
                right_cell = _Z[j + 1] + 1
            else:
                right_cell = _Z[j + 1]
            range_cell = np.arange(_Z[j], right_cell)  # The range of the cell
            _Q[j] = np.average(range_cell, weights=histORGN[_Z[j]:right_cell])  # Average calculation per boundary

            # mat is a matrix that is initialized in T / F. any value that satisfies the two conditions will get T, otherwise -F
            mat = np.logical_and(imOrig >= _Z[j], imOrig < right_cell)
            _newImg[mat] = _Q[j]  # Where there is a T we will update the new value

        imOr = imOrig / 255.0  # normalize
        imNew = _newImg / 255.0  # normalize
        MSE = np.sqrt(np.sum(np.square(np.subtract(imNew, imOr)))) / imOr.size  # According to MSE's formula
        MSE_lst.append(MSE)

        if flagRGB:  # RGB
            _newImg = _newImg / 255.0  # normalize
            imgYIQ[:, :, 0] = _newImg
            _newImg = transformYIQ2RGB(imgYIQ)  # Convert back to RGB
        quantized_lst.append(_newImg)  # add to quantized_lst

        _Z, _Q = change_position_boundary(_Z, _Q)  # each boundary become to be a middle of 2 means
        if len(MSE_lst) >= 2:
            if np.abs(MSE_lst[-1] - MSE_lst[-2]) <= 0.000001:
                break

    return quantized_lst, MSE_lst


def matrix() -> np.ndarray:
    """

    :return: matrix for formula: yiq = matrix * rgb
    """
    matrix_rgb = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
    return matrix_rgb


def change_position_boundary(_Z: np.ndarray, _Q: np.ndarray) -> (List[np.ndarray], List[np.ndarray]):
    """

    :param _Z: Represents the boundaries
    :param _Q: Represents the values of boundaries
    :return: _Z, _Q
    """
    for b in range(1, len(_Z) - 1):  # b is boundary
        _Z[b] = (_Q[b - 1] + _Q[b]) / 2  # # Average between boundaries
    return _Z, _Q
