import numpy as np
import torch
import matplotlib.pyplot as plt


def construct_measurement_matrix(M, N):
    Phi = np.random.randn(M, N)
    Q, R = np.linalg.qr(Phi)
    return Q

def gaussian2D(size, sigma_x=1, sigma_y=1):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)

    x, y = np.meshgrid(x, y)
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
        + y**2/(2*sigma_y**2))))
    return z

def image_to_array():
    image = plt.imread('brain_small.jpeg')
    image = np.mean(image, axis=2)
    return image


def generate_data(batch_size, m, n):
    A = construct_measurement_matrix(m, n)
    A = A.astype(np.float32)
    A = torch.from_numpy(A)
    image_size = int(n**0.5)
    x_gt_list = []
    for i in range(batch_size):
        #image = gaussian2D(image_size, sigma_x=0.3, sigma_y=0.7)
        image = image_to_array()
        image = image.reshape(n, 1)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        x_gt_list.append(image)
    x_gt = torch.stack(x_gt_list)

    b_list = []
    for x_per_batch in x_gt:
        b_per_batch = torch.matmul(A, x_per_batch)
        b_list.append(b_per_batch)

    b = torch.stack(b_list)
    return A, b, x_gt
