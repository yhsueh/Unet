import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*88*88, 120)  # 88*88 from image dimension
                                             # image size reduced after convolution and max pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
input = torch.randn(1,1,100,100)
print(net)

# Learnable parameters
params = list(net.parameters())
for i in params:
	print(i.size())

# Backprop with random gradients
'''
net.zero_grad()
out.backward(torch.randn(1,10))
'''

# Loss function
'''
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
'''

# Backprop
'''
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv2.bias.grad before backward')
print(net.conv2.bias.grad)

loss.backward()

print('conv2.bias.grad after backward')
print(net.conv2.bias.grad)
'''

# Create optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# Training
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()