import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from os.path import join
import os
import torch
from model2 import DeepBALM
import matplotlib.pyplot as plt
from datamaker import generate_data
from tqdm import tqdm

cs_ratio = 0.3
image_size = 60
n = image_size*image_size
m = int(cs_ratio * n)
batch_size = 1
dataset_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = DeepBALM(m, n, num_layer=10)
model.to(device)

A, observations, trainset = generate_data(dataset_size, m, n)
X = trainset.reshape((dataset_size, n))
Y = observations.reshape((dataset_size, m))
Q_init = X.T @ Y @ np.linalg.inv(Y.T @ Y)
# view dataset
'''
for i in range(dataset_size):
    x0 = Q_init @ observations[i]
    x_gt = trainset[i]
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x0.numpy().reshape((image_size, image_size)))
    axes[1].imshow(x_gt.numpy().reshape((image_size, image_size)))
    plt.show()
'''

A = A.to(device)
observations = observations.to(device)
trainset = trainset.to(device)
Q_init = Q_init.to(device)
print("###################Experiment settings#########################")
print(f"CS Ratio:{cs_ratio}")
print(f"A:{A.shape}, x:{trainset.shape}, b:{observations.shape}")
print("###############################################################")


x_init = torch.rand(batch_size, n, 1).to(device)
y_init = (10000*(torch.rand(batch_size, m, 1)-0.5)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_history = []
for epoch in range(500):
    epoch_loss = []
    for i in range(0, dataset_size, batch_size):
        x_gt = trainset[i:i+batch_size]
        b = observations[i:i+batch_size]
        x_init = Q_init @ b
        x_out, y_out, _ = model(x_init, y_init, A, b)
        loss = torch.mean(torch.pow(x_gt - x_out, 2))
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_history.append(np.mean(epoch_loss))
    print(f"Running epoch {epoch} with loss {loss_history[-1]}")
#torch.save(model.state_dict(), "gaussain_d100_e2000.pth")
plt.plot(loss_history)
plt.show()


# test
#model.load_state_dict(torch.load("gaussain_d100_e2000.pth", weights_only=True))
model.eval()
for i in range(dataset_size):
    x_gt, b = trainset[i], observations[i]
    image_gt = x_gt.to('cpu').numpy().reshape((image_size, image_size))
    x_init = (Q_init @ b).reshape((1,-1,1))
    x_out, y_out, x_list = model(x_init, y_init, A, b)
    fig, axes = plt.subplots(3, 4)
    for i, (x, ax) in enumerate(zip(x_list, axes.flatten())):
        image = x.detach().to('cpu').numpy()
        image = image[0].reshape((image_size, image_size))
        ax.imshow(image, cmap="gray")
        ax.set_title(f"{i} layer reconstruction")
    axes.flatten()[-1].imshow(image_gt, cmap="gray")
    axes.flatten()[-1].set_title("Ground Truth")
    plt.show()
