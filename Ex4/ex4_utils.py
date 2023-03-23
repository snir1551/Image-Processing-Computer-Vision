import cv2
from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt




def disparity_helper(img_l: np.ndarray,img_r: np.ndarray, disp_range: (int, int), k_size: int) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    """
    help function for disparitySSD and disparityNC functions

       img_l: Left image
       img_r: Right image
       disp_range: Minimum and Maximum disparity range. Ex. (10,80)
       k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

       return: disparity_map, mean_l, mean_r, height, width, normalize_left, normalize_right
       """

    height, width = img_r.shape
    disparity_map = np.zeros((height, width, disp_range[1]))

    mean_l = np.zeros((height, width))
    mean_r = np.zeros((height, width))

    filters.uniform_filter(img_l, k_size, mean_l) #use uniform_filter for calculate average
    filters.uniform_filter(img_r, k_size, mean_r) #use uniform_filter for calculate average

    #normalize img
    normalize_left = img_l - mean_l
    normalize_right = img_r - mean_r

    return disparity_map, mean_l, mean_r, height, width, normalize_left, normalize_right


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    disparity_map, left, right, height, width, normalize_left, normalize_right = disparity_helper(img_l, img_r, disp_range, k_size)

    for i in range(disp_range[1]):
        shiftImage = np.roll(normalize_right, i)
        filters.uniform_filter(normalize_left * shiftImage, k_size, disparity_map[:, :, i])
        disparity_map[:, :, i] = disparity_map[:, :, i] ** 2

    r = np.argmax(disparity_map, axis=2)
    return r



def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    disparity_map, left, right, height, width, normalize_left, normalize_right = disparity_helper(img_l, img_r, disp_range,
                                                                                           k_size)
    sigmaLeft = np.zeros((height, width))
    sigmaRight = np.zeros((height, width))
    sigma = np.zeros((height, width))

    filters.uniform_filter(normalize_left * normalize_left, k_size, sigmaLeft)

    for i in range(disp_range[1]):
        shiftImage = np.roll(normalize_right, i - disp_range[0])
        filters.uniform_filter(normalize_left * shiftImage, k_size, sigma)
        filters.uniform_filter(shiftImage * shiftImage, k_size, sigmaRight)
        sqrt = np.sqrt(sigmaRight * sigmaLeft)
        disparity_map[:, :, i] = sigma / sqrt

    a = np.argmax(disparity_map, axis=2)
    return a


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """

    vec = np.zeros((src_pnt.shape[0] * 2, 9))
    for z in range(src_pnt.shape[0]):
        vec[2 * z] = np.array([src_pnt[z][0], src_pnt[z][1], 1, 0, 0, 0, -dst_pnt[z][0] * src_pnt[z][0], -dst_pnt[z][0] * src_pnt[z][1],-dst_pnt[z][0]])
        vec[2 * z + 1] = np.array([0, 0, 0, src_pnt[z][0], src_pnt[z][1], 1, -dst_pnt[z][1] * src_pnt[z][0], -dst_pnt[z][1] * src_pnt[z][1],-dst_pnt[z][1]])

    a, b, h = np.linalg.svd(vec)
    homograph = h[-1].reshape(3, 3)
    homograph = homograph / homograph[2][2]

    one = np.ones(src_pnt.shape[0]).reshape(src_pnt.shape[0], 1) # take the err
    src_points = np.concatenate((src_pnt, one), axis=1).T
    dst_points = np.concatenate((dst_pnt, one), axis=1).T
    differ = homograph.dot(src_points) # warp
    differ = differ / differ[2, :]      # normalize
    differ = differ - dst_points
    err = np.sqrt(np.sum(differ ** 2))

    return homograph, err



def getEquation(point1, point2):
    """
      point 1:
      point 2:

      return: slope , z
      """

    # y1-y2
    upper = int(point1[1]) - int(point2[1])   # calculate
    # x1-x2
    lower = int(point1[0]) - int(point2[0])    # calculate
    if lower != 0:
        slope = upper / lower
        z = -slope * int(point1[0]) + int(point1[1]) #calculate
        return slope, z
    else:
        return 0, 0


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######
    src1 = []
    figure = plt.figure()  # take four points

    def onclick_2(event):
        xdata = event.xdata
        ydata = event.ydata
        print("Loc: {:.0f},{:.0f}".format(xdata, ydata))

        plt.plot(xdata, ydata, '*r')
        src1.append([xdata, ydata])

        if len(src1) == 4:
            plt.close()
        plt.show()

    figure = figure.canvas.mpl_connect('button_press_event', onclick_2)  # display image 2
    plt.imshow(src_img)
    plt.show()
    src1 = np.array(src1)


    # find the small x values, in dst_p img
    minimumX1 = float("inf")
    minimumX2 = float("inf")
    minimumXrow1 = -1
    minimumXrow2 = -1

    for row in range(len(dst_p)):
        if dst_p[row][0] <= minimumX1:
            minimumX1 = dst_p[row][0]
            minimumXrow1 = row
    for row in range(len(dst_p)):
        if minimumX1 <= dst_p[row][0] <= minimumX2 and row != minimumXrow1:
            minimumX2 = dst_p[row][0]
            minimumXrow2 = row
    # find smaller y and get the upper left and lower left
    if (dst_p[minimumXrow1][1] < dst_p[minimumXrow2][1]):
        tl = minimumXrow1
        bl = minimumXrow2
    else:
        tl = minimumXrow2
        bl = minimumXrow1

    # find bigger y and get the upper right and lower right
    listt = [0, 1, 2, 3]
    listt.remove(tl)
    listt.remove(bl)
    if dst_p[listt[0]][1] > dst_p[listt[1]][1]:
        tr = listt[1]
        br = listt[0]
    else:
        tr = listt[0]
        br = listt[1]


    # find the small x values, in src1
    minx1_src = float("inf")
    minx2_src = float("inf")
    minxrow1_src = -1
    minxrow2_src = -1
    for row in range(len(src1)):
        if src1[row][0] <= minx1_src:
            minx1_src = src1[row][0]
            minxrow1_src = row
    for row in range(len(src1)):
        if minx1_src <= src1[row][0] <= minx2_src and row != minxrow1_src:
            minx2_src = src1[row][0]
            minxrow2_src = row
    #  find the small y value and get the upper left and lower left corners
    if (src1[minxrow1_src][1] < src1[minxrow2_src][1]):
        tl_src = minxrow1_src
        bl_src = minxrow2_src
    else:
        tl_src = minxrow2_src
        bl_src = minxrow1_src


    listt = [0, 1, 2, 3]
    listt.remove(tl_src)
    listt.remove(bl_src)
    if src1[listt[0]][1] > src1[listt[1]][1]:
        tr_src = listt[1]
        br_src = listt[0]
    else:
        tr_src = listt[0]
        br_src = listt[1]

    # make a new src
    src_p = np.zeros_like(src1)
    src_p[tl, :] = src1[tl_src, :]
    src_p[tr, :] = src1[tr_src, :]
    src_p[br, :] = src1[br_src, :]
    src_p[bl, :] = src1[bl_src, :]

    mask = np.zeros((dst_img.shape[0], dst_img.shape[1], 3))
    maximumX = max([dst_p[br][0], dst_p[tr][0]])
    minimumX = min([dst_p[bl][0], dst_p[tl][0]])
    maximumY = max([dst_p[br][1], dst_p[bl][1]])
    minimumY = min([dst_p[tr][1], dst_p[tl][1]])

    TL_TR_slope, TL_TR_b = getEquation((dst_p[tl][0], dst_p[tl][1]), (dst_p[tr][0], dst_p[tr][1]))
    BL_BR_slope, BL_BR_b = getEquation((dst_p[bl][0], dst_p[bl][1]), (dst_p[br][0], dst_p[br][1]))
    TL_BL_slope, TL_BL_b = getEquation((dst_p[tl][0], dst_p[tl][1]), (dst_p[bl][0], dst_p[bl][1]))
    TR_BR_slope, TR_BR_b = getEquation((dst_p[br][0], dst_p[br][1]), (dst_p[tr][0], dst_p[tr][1]))

    # run on the image
    for x in range(dst_img.shape[0]):
        for y in range(dst_img.shape[1]):
            if minimumX <= y <= maximumX and minimumY <= x <= maximumY:
                if TL_TR_slope * y + TL_TR_b <= x <= BL_BR_slope * y + BL_BR_b:
                    if TL_BL_slope != 0 and TR_BR_slope != 0:
                        if (x - TL_BL_b) / TL_BL_slope <= y <= ( x - TR_BR_b) / TR_BR_slope:
                            mask[x][y][0] = 1
                            mask[x][y][1] = 1
                            mask[x][y][2] = 1
                    elif TL_BL_slope != 0 and TR_BR_slope == 0:
                        if x <= dst_p[br][1] and (x - TL_BL_b) / TL_BL_slope <= y:
                            mask[x][y][0] = 1
                            mask[x][y][1] = 1
                            mask[x][y][2] = 1

                    elif TL_BL_slope == 0 and TR_BR_slope != 0:
                        if (x - TR_BR_b) / TR_BR_slope >= y and x >= dst_p[tl][0]:
                            mask[x][y][0] = 1
                            mask[x][y][1] = 1
                            mask[x][y][2] = 1

                    else:  # parallel
                        if dst_p[tl][0] <= x <= dst_p[tr][0]:
                            mask[x][y][0] = 1
                            mask[x][y][1] = 1
                            mask[x][y][2] = 1


    homograph, error = computeHomography(src_p, dst_p) # homgraphy
    srcOutput = cv2.warpPerspective(src_img, homograph, (dst_img.shape[1], dst_img.shape[0]))  # warp

    out = dst_img * (1 - mask) + srcOutput * (mask)   # link
    plt.imshow(out)
    plt.show()
