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
	model.load_state_dict(torch.load('./model/05_29_22 17_12_45.pt'))
	model.eval()

	with Image.open('./dataset/test_img/crop_22.png').convert('L') as im:
	    f, ax = plt.subplots(1,2)
	    ax[0].imshow(im, cmap='gray')

	    im = np.array(im).astype('float32')
	    im /= 255.0
	    torch_im = torch.from_numpy(im)
	    torch_im = torch.unsqueeze(torch_im, dim=0)
	    torch_im = torch.unsqueeze(torch_im, dim=0)

	    with torch.no_grad():
	        output = model(torch_im)

	    probs = F.softmax(output, dim=1).squeeze(dim=0)[0]

	    tf = transforms.Compose([
	    transforms.ToPILImage(),
	    transforms.Resize((576, 576), transforms.InterpolationMode.BICUBIC)
	    ])
	    
	    mask = np.array(tf(probs))
	    ax[1].imshow((mask>0), cmap='gray')
	    plt.show()

if __name__ == '__main__':
	main()