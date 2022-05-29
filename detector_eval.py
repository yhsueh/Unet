import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os
import datetime
import unet


def main():
	model = unet.Unet()
	model.load_state_dict(torch.load('.\\model\\05_29_22 14_03_02'))
	model.eval()

	with Image.open('.\\dataset\\crop_19.png').convert('L') as im:
		im = np.array(im).astype('float32')
		im /= 255.0

		torch_im = torch.from_numpy(im)
		torch_im = torch.unsqueeze(torch_im, dim=0)
		torch_im = torch.unsqueeze(torch_im, dim=0)

		with torch.no_grad():
			output = model(torch_im)

		softmax = torch.nn.Softmax2d()
		output = softmax(output)

		tf = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((576, 576), transforms.InterpolationMode.BICUBIC)
			])

		mask = tf(output[0,0,:,:])
		plt.imshow(mask, cmap='gray')
		plt.show()



if __name__ == '__main__':
	main()