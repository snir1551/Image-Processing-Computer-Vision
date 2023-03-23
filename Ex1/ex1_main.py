from ex1_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time


def histEqDemo(img_path: str, rep: int):
    print("check8")
    img = imReadAndConvert(img_path, rep)
    print("check9")
    imgeq, histOrg, histEq = hsitogramEqualize(img)
    print("check10")

    # Display cumsum
    cumsum = np.cumsum(histOrg)
    print("check11")
    cumsumEq = np.cumsum(histEq)
    print("check12")
    plt.gray()
    print("check13")
    plt.plot(range(256), cumsum, 'r')
    print("check14")
    plt.plot(range(256), cumsumEq, 'g')
    print("check15")

    # Display the images
    plt.figure()
    print("check16")
    plt.imshow(img)
    print("check17")

    plt.figure()
    print("check18")
    plt.imshow(imgeq)
    print("check19")
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()

    img_lst, err_lst = quantizeImage(img, 3, 20)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %f" % err_lst[0])
    print("Error last:\t %f" % err_lst[-1])

    plt.gray()
    plt.imshow(img_lst[0])
    plt.figure()
    plt.imshow(img_lst[-1])

    plt.figure()
    plt.plot(err_lst, 'r')
    plt.show()


def main():
    #print("ID:", myID())
    img_path = 'beach.jpg'

    # Basic read and display
    imDisplay(img_path, LOAD_GRAY_SCALE)
    print("check1")
    imDisplay(img_path, LOAD_RGB)
    print("check2")

    # Convert Color spaces
    img = imReadAndConvert(img_path, LOAD_RGB)
    print("check3")
    yiq_img = transformRGB2YIQ(img)
    print("check4")
    #print(yiq_img)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    print("check5")
    ax[1].imshow(yiq_img)
    print("check6")
    plt.show()
    print("check7")
    # Image histEq
    histEqDemo(img_path, LOAD_GRAY_SCALE)
    histEqDemo(img_path, LOAD_RGB)

    # Image Quantization



    quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB)

    # Gamma
    gammaDisplay(img_path, LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
