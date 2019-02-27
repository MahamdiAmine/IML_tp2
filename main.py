import cv2
import numpy as np
from skimage.util import random_noise
from scipy import ndimage
import matplotlib.pyplot as plt


def read_img(path):
    try:
        # return cv2.imread(path, 0)
        return cv2.imread(path)
    except IOError as e:
        print(e)
        return None


def display(path):
    img = read_img(path)
    image = cv2.resize(img, (720, 500))
    height, width = img.shape
    title = 'image' + ' width = ' + str(width) + ' height = ' + str(height)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def noises(path):
    img1 = cv2.imread(path)
    return img1, random_noise(img1, mode='gaussian', seed=None, clip=True), \
           random_noise(img1, mode='s&p', seed=None, clip=True)


def display_noises(path):
    img1, gauss1, sp1 = noises(path[0])
    img2, gauss2, sp2 = noises(path[1])
    images = [[img1, sp1, gauss1], [img2, sp2, gauss2]]
    titles = ["Original image", "s&p", "Gaussian noise"]
    fig = plt.figure()
    fig.suptitle("Adding noises :", fontsize=19, color='red')

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.title(titles[i % 2])
        if i < 3:
            plt.imshow(images[0][i])
        else:
            plt.imshow(images[1][i - 3])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def HP_filter(path):
    img1, gauss1, sp1 = noises(path)
    # A very simple and very narrow highpass filter
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    gauss_data1 = np.array(gauss1, dtype=float)
    sp_data1 = np.array(sp1, dtype=float)
    normal_data1 = np.array(img1, dtype=float)
    highpass_3x3_g = ndimage.convolve(gauss_data1, kernel)
    highpass_3x3_s = ndimage.convolve(sp_data1, kernel)
    highpass_3x3_n = ndimage.convolve(normal_data1, kernel)
    images = [highpass_3x3_n, highpass_3x3_s, highpass_3x3_g]
    titles = ["Original image", "s&p", "Gaussian noise"]
    fig = plt.figure()
    fig.suptitle("High pass filtering :", fontsize=19, color='red')
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def median_filter(path, param=4):
    img1, gauss1, sp1 = noises(path)
    kernel_3x3 = np.ones((3, 3), np.float32) / 9
    kernel_4x4 = np.ones((4, 4), np.float32) / 16
    kernel_5x5 = np.ones((5, 5), np.float32) / 25
    if param == 3:
        kernel = kernel_3x3
    elif param == 4:
        kernel = kernel_4x4
    elif param == 5:
        kernel = kernel_5x5
    data_n = cv2.filter2D(img1, -1, kernel)
    data_s = cv2.filter2D(sp1, -1, kernel)
    data_g = cv2.filter2D(gauss1, -1, kernel)

    images = [data_n, data_s, data_g]
    titles = ["Original image", "s&p", "Gaussian noise"]
    fig = plt.figure()
    title = "Median filter " + str(param) + "x" + str(param) + " :"
    fig.suptitle(title, fontsize=19, color='red')
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def LPF(path, param=4):
    img1, gauss1, sp1 = noises(path)
    kernel_3x3 = np.ones((3, 3), np.float32)
    kernel_4x4 = np.ones((4, 4), np.float32)
    kernel_5x5 = np.ones((5, 5), np.float32)
    if param == 3:
        kernel = kernel_3x3
    elif param == 4:
        kernel = kernel_4x4
    elif param == 5:
        kernel = kernel_5x5
    data_n = cv2.filter2D(img1, -1, kernel)
    data_s = cv2.filter2D(sp1, -1, kernel)
    data_g = cv2.filter2D(gauss1, -1, kernel)

    images = [data_n, data_s, data_g]
    titles = ["Original image", "s&p", "Gaussian noise"]
    fig = plt.figure()
    title = "Median filter " + str(param) + "x" + str(param) + " :"
    fig.suptitle(title, fontsize=19, color='red')
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def Amethyst_Contrast_Enhancer(path, alpha, beta):
    # alpha [1.0-3.0] ,beta  [0-100]
    if not (3.0 > alpha > 0 and 100 > beta > 0):
        print("alpha between [1.0-3.0] and beta must be between [0-100]")
        exit(88)
    image = cv2.imread(path)
    new_image = np.zeros(image.shape, image.dtype)
    # Initialize values
    print(' Basic Linear Transforms ')
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)

    tiltle1 = "Original Image" + "alpha = " + str(alpha) + " beta = " + str(beta) + " :"
    tiltle2 = "New Image" + "alpha = " + str(alpha) + " beta = " + str(beta) + " :"
    cv2.imshow(tiltle1, image)
    cv2.imshow(tiltle2, new_image)
    # Wait until user press some key
    cv2.waitKey()


if __name__ == '__main__':
    path = ['./data/trui.png', './data/cameraman.jpg']
    # display_noises(path)
    # HP_filter(path[0])
    # HP_filter(path[1])
    # median_filter(path[0], 5)
    # median_filter(path[0], 3)
    # median_filter(path[1], 5)
    # LPF(path[1], 3)
    Amethyst_Contrast_Enhancer(path[0], 2, 85)  # alpha [1.0-3.0] ,beta  [0-100]
