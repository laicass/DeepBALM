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


image_size = 30
m = 2000
n = image_size*image_size
batch_size = 1
dataset_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = DeepBALM(m, n, num_layer=10)
model.to(device)

A, observations, trainset = generate_data(dataset_size, m, n)

# view dataset
'''
for x in x_gt.numpy():
    x = x.reshape((image_size, image_size))
    plt.imshow(x)
    plt.show()
'''

A = A.to(device)
observations = observations.to(device)
trainset = trainset.to(device)
print(trainset.shape, observations.shape)

x_init = torch.rand(batch_size, n, 1).to(device)
y_init = (10000*(torch.rand(batch_size, m, 1)-0.5)).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_history = []


for epoch in range(1000):
    for x_gt, b in zip(trainset, observations):
        x_out, y_out, _ = model(x_init, y_init, A, b)
        loss = torch.mean(torch.pow(x_gt - x_out, 2))
        loss_history.append(loss.item())
        print(f"Running epoch {epoch} with loss {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


output = x_out.detach().to('cpu').numpy()
image_pred = output[0].reshape((image_size, image_size))
image_gt = trainset.to('cpu').numpy()[0].reshape((image_size, image_size))
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image_gt, cmap='gray')
axes[0].set_title("Ground Truth")
axes[1].imshow(image_pred, cmap='gray')
axes[1].set_title("Reconstructed")

plt.show()
plt.plot(loss_history)
plt.show()

#np.save("y_init", y.to('cpu').numpy())

# test
x_init = torch.zeros(batch_size, n, 1).to(device)
test_b = observations[0]
with torch.no_grad():
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
