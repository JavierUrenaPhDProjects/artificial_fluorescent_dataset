import sys
import numpy as np
import cv2
import os
import shutil
import pandas as pd
from tqdm import tqdm


def show_image(img):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    imS = cv2.resize(img, (1920, 1080))  # Resize image
    cv2.imshow("Image", imS)  # Show image
    print('Press Esc key to kill')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def f(x, y, A, sigma_x, sigma_y, theta):
    # Gaussian function obtained from https://www.astro.rug.nl/~vogelaar/Gaussians2D/2dgaussians.html
    theta = np.radians(theta)
    sigx2 = sigma_x ** 2
    sigy2 = sigma_y ** 2
    a = np.cos(theta) ** 2 / (2 * sigx2) + np.sin(theta) ** 2 / (2 * sigy2)
    b = np.sin(theta) ** 2 / (2 * sigx2) + np.cos(theta) ** 2 / (2 * sigy2)
    c = np.sin(2 * theta) / (4 * sigx2) - np.sin(2 * theta) / (4 * sigy2)

    expo = -a * x ** 2 - b * y ** 2 - 2 * c * x * y
    return A * np.exp(expo)


def random_cell():
    N = np.random.randint(100, 150)
    x = np.linspace(-5, 5, N)
    y = np.linspace(-3, 3, N)

    theta = np.random.randint(0, 360)  # deg
    sigx = np.random.uniform(0.5, 1)
    sigy = np.random.uniform(0.5, 1)
    A = 1
    Xg, Yg = np.meshgrid(x, y)

    Z = f(Xg, Yg, A, sigx, sigy, theta)

    return Z, N


def add_cells(img, n_cells):
    H, W, _ = img.shape
    centroids = []
    for _ in range(n_cells):
        try:
            roi = [np.random.randint(H), np.random.randint(W)]
            cell, size = random_cell()
            patch = img[roi[0]:roi[0] + size, roi[1]:roi[1] + size, :]
            max_val_patch = patch.max()
            max_val_cell = 255 - max_val_patch
            cell = cell * max_val_cell
            patch[:, :, 1] = patch[:, :, 1] + cell
            img[roi[0]:roi[0] + size, roi[1]:roi[1] + size, :] = patch

            center = [roi[0] + int(size / 2), roi[1] + int(size / 2)]
            centroids.append(center)
        except:
            pass

    return img, centroids


def put_marks(img, centroids):
    markerType = cv2.MARKER_STAR
    color = (0, 0, 255)
    markerSize = 15
    thickness = 2

    for c in centroids:
        y, x = c
        cv2.drawMarker(img, (x, y), color, markerType, markerSize, thickness)

    return img


def create_data_image(background_path, n_cells):
    backgrounds_paths = [os.path.join(background_path, file) for file in os.listdir(background_path)]
    bg_path = np.random.choice(backgrounds_paths)

    bg = cv2.imread(bg_path)
    img, centroids = add_cells(bg, n_cells)

    return img, centroids


def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'images'))
    else:
        print(f'The folder {path} already exists. It has {len(os.listdir(path))} elements.')
        print('Do you want to remove its content? y/n')
        if input('') == 'y':
            shutil.rmtree(path)
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'images'))
        else:
            sys.exit()


def create_dataset(background_path, output_path, N=100, max_n=2000):
    check_folder(output_path)
    images_path = os.path.join(output_path, 'images')

    range_list = np.linspace(0, max_n, N, dtype=int)
    id_ = 1
    dataset_dic = {'file': [], 'gt': [], 'centroids': []}

    for n in tqdm(range_list):
        image, centroids = create_data_image(background_path, n)
        img_path = os.path.join(images_path, f'{id_}.png')
        cv2.imwrite(img_path, image)

        dataset_dic['file'].append(f'{id_}.png')
        dataset_dic['gt'].append(len(centroids))
        dataset_dic['centroids'].append(centroids)

        id_ += 1

    df = pd.DataFrame(dataset_dic)
    df.to_csv(os.path.join(output_path, 'artificial_fluorescent_dataset.csv'), index=False)


def test():
    img_2, centroids = create_data_image('DATASET/Background', 500)
    show_image(img_2)
    show_image(put_marks(img_2, centroids))


if __name__ == '__main__':
    test()
