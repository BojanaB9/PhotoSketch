import numpy as np
import cv2
from skimage import io, color, filters, transform
from scipy import signal
import matplotlib.pyplot as plt


def gen_stroke_map(img, kernel_size, stroke_width=0, num_of_directions=8, smooth_kernel="gauss", gradient_method=0):
    height = img.shape[0]
    width = img.shape[1]

    if smooth_kernel == "gauss":
        smooth_im = filters.gaussian(img, sigma=np.sqrt(2))
    else:
        smooth_im = filters.median(img)

    if not gradient_method:
        im_x = np.zeros_like(smooth_im)
        diff_x = img[:, 1:width] - img[:, 0:width - 1]
        im_x[:, 0:width - 1] = diff_x
        im_y = np.zeros_like(img)
        diff_y = img[1:height, :] - img[0:height - 1, :]
        im_y[0:height - 1, :] = diff_y
        g = np.sqrt(np.square(im_x) + np.square(im_y))
    else:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        g = np.sqrt(np.square(sobelx) + np.square(sobely))

    basic_ker = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1))
    basic_ker[kernel_size + 1, :] = 1

    res_map = np.zeros((height, width, num_of_directions))
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        res_map[:, :, d] = signal.convolve2d(g, ker, mode='same')
    max_pixel_indices_map = np.argmax(res_map, axis=2)

    c = np.zeros_like(res_map)
    for d in range(num_of_directions):
        c[:, :, d] = g * (max_pixel_indices_map == d)

    if not stroke_width:
        for w in range(1, stroke_width + 1):
            if (kernel_size + 1 - w) >= 0:
                basic_ker[kernel_size + 1 - w, :] = 1
            if (kernel_size + 1 + w) < (kernel_size * 2 + 1):
                basic_ker[kernel_size + 1 + w, :] = 1

    s_tag_sep = np.zeros_like(c)
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        s_tag_sep[:, :, d] = signal.convolve2d(c[:, :, d], ker, mode='same')
    s_tag = np.sum(s_tag_sep, axis=2)

    s_tag_normalized = (s_tag - np.min(s_tag.ravel())) / (np.max(s_tag.ravel()) - np.min(s_tag.ravel()))

    s = 1 - s_tag_normalized
    return s


def gen_pencil_drawing(img, kernel_size=8, stroke_width=2, num_of_directions=8, smooth_kernel="median",
                       gradient_method=0, stroke_darkness=2):
    im = img
    S = gen_stroke_map(im, kernel_size, stroke_width=stroke_width, num_of_directions=num_of_directions,
                       smooth_kernel=smooth_kernel, gradient_method=gradient_method)
    S = np.power(S, stroke_darkness)

    return S


def main():
    img = io.imread('house.png')

    if img.shape[2] == 4:
        img = color.rgba2rgb(img)
    im = color.rgb2gray(img)

    res = gen_pencil_drawing(im, kernel_size=8, stroke_width=4, num_of_directions=8, smooth_kernel="gauss",
                             gradient_method=1, stroke_darkness=2)

    plt.rcParams['figure.figsize'] = [16, 10]
    plt.subplot(1, 2, 1).set_title("Оригинална слика")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 2, 2).set_title("Скица")
    plt.imshow(res, cmap='gray')
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
