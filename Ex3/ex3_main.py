from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float), img_2.astype(np.float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv, 0))
    print(np.mean(uv, 0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")

    # im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))
    st = time.time()
    ans = opticalFlowPyrLK(im1.astype(np.float), im2.astype(np.float), 4, 20, 5)
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    pts = np.array([])
    uv = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uv = np.append(uv, ans[i][j][0])
                uv = np.append(uv, ans[i][j][1])
                pts = np.append(pts, j)
                pts = np.append(pts, i)
    pts = pts.reshape(int(pts.shape[0] / 2), 2)
    uv = uv.reshape(int(uv.shape[0] / 2), 2)
    print(np.median(uv, 0))
    print(np.mean(uv, 0))
    displayOpticalFlow(im2, pts, uv)


# saveOpticalFlow("opticalFlowPyrLKCompare",im2, pts, uv)

def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """

    print("Compare LK & Hierarchical LK")

    # im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -0.2],
                  [0, 1, -0.1],
                  [0, 0, 1]], dtype=np.float)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

    pts, uv = opticalFlow(im1.astype(np.float), im2.astype(np.float), step_size=20, win_size=5)

    ans = opticalFlowPyrLK(im1.astype(np.float), im2.astype(np.float), 4, 20, 5)
    ptspyr = np.array([])
    uvpyr = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uvpyr = np.append(uvpyr, ans[i][j][0])
                uvpyr = np.append(uvpyr, ans[i][j][1])
                ptspyr = np.append(ptspyr, j)
                ptspyr = np.append(ptspyr, i)
    ptspyr = ptspyr.reshape(int(ptspyr.shape[0] / 2), 2)
    uvpyr = uvpyr.reshape(int(uvpyr.shape[0] / 2), 2)
    if len(im2.shape) == 2:
        f, ax = plt.subplots(1, 3)
        ax[0].set_title('opticalFlow')
        ax[0].imshow(im2, cmap="gray")
        ax[0].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        ax[1].set_title('opticalFlowPyrLK')
        ax[1].imshow(im2, cmap="gray")
        ax[1].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='r')
        ax[2].set_title('comp')
        ax[2].imshow(im2, cmap="gray")
        ax[2].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        ax[2].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='y')
        plt.show()

    else:
        f, ax = plt.subplots(1, 3)
        ax[0].set_title('regular LK')
        ax[0].imshow(im2)
        ax[0].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        ax[1].set_title('Pyr LK')
        ax[1].imshow(im2)
        ax[1].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='r')
        ax[2].set_title('overlap')
        ax[2].imshow(im2)
        ax[2].quiver(pts[:, 0], pts[:, 1], uv[:, 0], uv[:, 1], color='r')
        ax[2].quiver(ptspyr[:, 0], ptspyr[:, 1], uvpyr[:, 0], uvpyr[:, 1], color='y')
        plt.show()


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK_Tester(img_path):  # 3.1

    print("findTranslationLK_Tester")

    # img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.1],
                  [0, 1, .4],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, (img_1.shape[1], img_1.shape[0]))
    cv2.imwrite('ImgTest/imTransA1.jpg', img_2)
    st = time.time()
    mat = findTranslationLK(img_1, img_2)
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    new = cv2.warpPerspective(img_1, mat, (img_1.shape[1], img_1.shape[0]))
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('image2 given transf')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('image2 found transf')
    ax[1].imshow(new, cmap='gray')

    ax[2].set_title('diff')
    ax[2].imshow(img_2 - new, cmap='gray')

    plt.show()
    print("mse= ", MSE(new, img_2))


def findRigidLK_Tester(img_path):  # 3.2

    print("findRigidLK_Tester")

    # img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    t1 = np.array([[1, 0, -.2],
                   [0, 1, .2],
                   [0, 0, 1]], dtype=np.float)
    theta = 0.01
    t2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]], dtype=np.float)

    t = t1 @ t2
    img_2 = cv2.warpPerspective(img_1, t, (img_1.shape[1], img_1.shape[0]))
    cv2.imwrite('ImgTest/imRigidA1.jpg', img_2)

    st = time.time()
    mat = findRigidLK(img_1, img_2)
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    new = cv2.warpPerspective(img_1, mat, (img_1.shape[1], img_1.shape[0]))
    if len(img_2.shape) == 2:
        f, ax = plt.subplots(1, 3)
        ax[0].set_title('image2 given transf')
        ax[0].imshow(img_2, cmap='gray')

        ax[1].set_title('image2 found transf')
        ax[1].imshow(new, cmap='gray')

        ax[2].set_title('diff')
        ax[2].imshow(img_2 - new, cmap='gray')

        plt.show()
    else:
        f, ax = plt.subplots(1, 3)
        ax[0].set_title('image2 given transf')
        ax[0].imshow(img_2)

        ax[1].set_title('image2 found transf')
        ax[1].imshow(new)

        ax[2].set_title('diff')
        ax[2].imshow(img_2 - new)

        plt.show()
    print("mse= ", MSE(new, img_2))


def findTranslationCorr_Tester(img_path):  # 3.3
    print("findTranslationCorr_Tester")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -5],
                  [0, 1, 7],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, (img_1.shape[1], img_1.shape[0]))
    cv2.imwrite('ImgTest/imTransA2.jpg', img_2)

    st = time.time()
    mat = findTranslationCorr(img_1.astype(np.float), img_2.astype(np.float))
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    new = cv2.warpPerspective(img_1, mat, (img_1.shape[1], img_1.shape[0]))

    f, ax = plt.subplots(1, 3)
    ax[0].set_title('image2 given transf')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('image2 found transf')
    ax[1].imshow(new, cmap='gray')

    ax[2].set_title('diff')
    ax[2].imshow(img_2 - new, cmap='gray')

    plt.show()
    print("mse= ", MSE(new, img_2))


def findRigidCorr_Tester(img_path):  # 3.4
    print("findRigidCorr_Tester")
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)

    theta = 0.1
    t = np.array([[np.cos(theta), -np.sin(theta), -10],
                  [np.sin(theta), np.cos(theta), -1],
                  [0, 0, 1]], dtype=np.float)

    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    cv2.imwrite('ImgTest/imRigidB2.jpg', img_2)
    st = time.time()
    mat = findRigidCorr(img_1.astype(np.float), img_2.astype(np.float))
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    # print("mat\n", mat, "\nt\n", t)
    new = cv2.warpPerspective(img_1, mat, img_1.shape[::-1])

    f, ax = plt.subplots(1, 3)
    ax[0].set_title('image2 given trans')
    ax[0].imshow(img_2, cmap='gray')

    ax[1].set_title('image2 found transf')
    ax[1].imshow(new, cmap='gray')

    ax[2].set_title('diff')
    ax[2].imshow(img_2 - new, cmap='gray')

    plt.show()
    print("mse= ", MSE(new, img_2))


##################
def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """

    print("Image Warping Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)

    theta = 0.01
    t = np.array([[np.cos(theta), -np.sin(theta), -0.5],
                  [np.sin(theta), np.cos(theta), 0.5],
                  [0, 0, 1]], dtype=np.float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])

    st = time.time()
    im2 = warpImages(img_1.astype(np.float), img_2.astype(np.float), t)
    et = time.time()
    print("Time: {:.4f}".format(et - st))

    f, ax = plt.subplots(1, 3)
    ax[0].set_title('my rewarp')
    ax[0].imshow(im2)

    ax[1].set_title('cv2 warp')
    ax[1].imshow(img_1)

    ax[2].set_title('diff')
    ax[2].imshow(img_2 - im2)
    plt.show()


#    print("MSE: {}".format(MSE(img_2, im2)))
#    print("Max Error: {}".format(np.abs(img_2 - im2).max()))

def MSE(a: np.ndarray, b: np.ndarray) -> float:
    return np.square(a - b).mean()


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    #print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    compareLK(img_path)

    # ----Tests----#
    my_img_path = 'ImgTest/ball.jpg'
    findTranslationLK_Tester(my_img_path)
    findRigidLK_Tester(my_img_path)
    findTranslationCorr_Tester(my_img_path)
    findRigidCorr_Tester(my_img_path)
    # ----Tests----#
    imageWarpingDemo(my_img_path)

    imageWarpingDemo(img_path)

    pyrGaussianDemo('input/pyr_bit.jpg')
    pyrLaplacianDemo('input/pyr_bit.jpg')
    blendDemo()


if __name__ == '__main__':
    main()
