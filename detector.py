import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os
import unet

class Detector:
    def __init__(self): 
        self.model = unet.Encoder()
        self.configuration()

    def configuration(self):
        ### hyperparameters ###
        # defining the number of epochs
        self.n_epochs = 10
        
        # define batch size
        self.batch_size = 10
       
        # empty list to store training losses
        self.train_losses = []
        
        # empty list to store validation losses
        self.val_losses = []

        ### optimizer ###
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01)

        ### criterion ###
        self.criterion = nn.CrossEntropyLoss()

    def load_image(self, path):
        img_path = os.path.join(path, "img")
        mask_path = os.path.join(path, "mask")
        
        #train_img = self._grab_image(img_path)
        train_mask = self._grab_image(mask_path, True)

        self.train_x = torch.from_numpy(train_img)
        self.train_y = torch.from_numpy(train_mask)

    def _grab_image(self, directory, mask=False):
        imgs = []
        directory = os.path.join(directory + '\\*.png')
        for infile in glob.glob(directory):
                with Image.open(infile).convert('L') as im:
                    if not mask:
                        im = np.array(im).astype('float32')
                        im /= 255.0
                        imgs.append(im)
                    else:
                        mask_class_one = np.array(im).astype(bool)
                        mask_class_two = ~mask_class_one
                        imgs.append([mask_class_one.astype('float32'), mask_class_two.astype('float32')])
        imgs = np.array(imgs)
        return imgs

    def train(self):
        print('Begin training')

        for epoch in range(self.n_epochs):
            loss_train = 0
            # randomly generate subsample batch
            permutation = torch.randperm(self.train_x.size()[0])

            for i in range(0, self.train_x.size()[0], self.batch_size):
                ### Clearing the graidents of the model aprameters ###
                self.optimizer.zero_grad()

                ### Getting training batches ###
                indicies = permutation[i:i+self.batch_size]
                batch_x = self.train_x[indicies]
                batch_y = self.train_y[indicies]
                print('batch_x: ', batch_x.shape) # tensor size: 10, 400, 400
                print('batch_y: ', batch_y.shape)

                batch_x = torch.unsqueeze(batch_x, dim = 1)
                batch_y = torch.unsqueeze(batch_y, dim = 1)
                ### Model Prediction ###
                predictions = self.model(batch_x)                

                ### Crop mask in order to match it with the prediction ###
                batch_y = batch_y[:, :, 20:380, 20:380]

                #print('prediction size:', predictions.shape)
                #print('batch_y', batch_y.shape)

                ### Compute the training loss ###
                loss_train = self.criterion(predictions, batch_y)
                self.train_losses.append(loss_train)

                ### Backprop and update weights ###
                loss_train.backward()
                self.optimizer.step()
            print('Epoch: {}, Loss: {}\n'.format(epoch, loss_train))

def main():
    detector = Detector()
    detector.load_image('.\\dataset')
    detector.train()

if __name__ == '__main__':
    main()