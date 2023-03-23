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

import cv2
import numpy as np
from ex1_utils import LOAD_GRAY_SCALE


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    img_for_gamma = cv2.imread(img_path)  # load image
    if rep == LOAD_GRAY_SCALE:  # GRAYSCALE image
        img_for_gamma = cv2.cvtColor(img_for_gamma, cv2.COLOR_BGR2GRAY)  # convert from BGR to GRAYSCALE
    title_window = 'Gamma Correction'  # the title of the window
    trackbar_name = 'Gamma'  # the trackbar name
    cv2.namedWindow(title_window)
    cv2.createTrackbar(trackbar_name, title_window, 0, 100, on_trackbar)
    while True:
        gamma = cv2.getTrackbarPos(trackbar_name, title_window)  # trackbar position
        gamma = gamma / 100 * (2 - 0.01)  # The slider’s values should be from 0 to 2 with resolution 0.01
        gamma = 0.01 if gamma == 0 else gamma  # if 0 should be 0.01
        newImg = gamma_correction(img_for_gamma, gamma)  # game correction
        cv2.imshow(title_window, newImg)  # show img
        k = cv2.waitKey(1000)
        if k == 27:  # esc button
            break
        if cv2.getWindowProperty(title_window, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
        Gamma correction
        :param image: the original image
        :param gamma: the gamma number
        :return: the new image after the gamma operation
        """
    gamma_for_formula = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_for_formula) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def on_trackbar(x: int):  # do nothing
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
