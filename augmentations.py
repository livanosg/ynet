import numpy as np
from cv2.cv2 import GaussianBlur, filter2D, warpAffine, getRotationMatrix2D, flip


def rotate(input_image, label):
    angle = np.random.randint(-45, 45)
    rows, cols = input_image.shape
    m = getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
    return warpAffine(input_image, m, (cols, rows)), warpAffine(label, m, (cols, rows))


def flips(input_image, label):
    flip_flag = np.random.randint(-1, 2)
    return flip(input_image, flip_flag), flip(label, flip_flag)


# noinspection DuplicatedCode
def s_n_p(input_image, label):
    p, b = 0.5, 0.0005
    max_val = np.max(input_image)
    num_salt = np.ceil(b * input_image.size * p)
    coords = tuple([np.random.randint(0, dim - 1, int(num_salt)) for dim in input_image.shape])
    input_image[coords] = max_val
    num_pepper = np.ceil(b * input_image.size * (1. - p))
    coords = tuple([np.random.randint(0, dim - 1, int(num_pepper)) for dim in input_image.shape])
    input_image[coords] = 0
    return input_image, label


def sharp(input_image, label):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return filter2D(input_image, -1, kernel), label


def gaussian_blur(input_image, label):
    return GaussianBlur(input_image, (3, 3), sigmaX=1.5, sigmaY=1.5), label


def contrast(input_image, label):
    contrast_factor = np.random.rand() * 2.
    image_mean = np.mean(input_image)
    image_contr = (input_image - image_mean) * contrast_factor + image_mean
    return image_contr, label


def random_translation(input_image, label):
    x = np.random.random_integers(-80, 80)
    y = np.random.random_integers(-80, 80)
    m = np.float32([[1, 0, x], [0, 1, y]])
    rows, cols, = input_image.shape
    return warpAffine(input_image, m, (cols, rows)), warpAffine(label, m, (cols, rows))


def augmentations(dcm_image, grd_image):
    # Random choice of augmentation method
    all_processes = [rotate, flips, random_translation, s_n_p, sharp, gaussian_blur, contrast]
    augm = np.random.choice(all_processes)
    dcm_image, grd_image = augm(dcm_image, grd_image)
    if np.random.random() < 0.5:  # Data augmentation:
        all_processes.pop(all_processes.index(augm))  # pop patients from list
        augm = np.random.choice(all_processes)
        dcm_image, grd_image = augm(dcm_image, grd_image)
    return augm(dcm_image, grd_image)
