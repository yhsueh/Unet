import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import os
import datetime
import unet

global verbose
verbose = False

class Detector:
    def __init__(self): 
        self.model = unet.Unet()
        self.configuration()

    def configuration(self):
        ### hyperparameters ###
        # defining the number of epochs
        self.n_epochs = 1
        
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
        
        train_img = self._grab_image(img_path)
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
                        # class one: defects; class two: background
                        mask_class_one = np.array(im).astype(bool)
                        mask_class_two = ~mask_class_one

                        '''
                        f, ax = plt.subplots(1,2)
                        print('mask_one', mask_class_one.astype('float32'))
                        print('mask_two', mask_class_two.astype('float32'))
                        ax[0].imshow(mask_class_one, cmap='gray')
                        ax[1].imshow(mask_class_two, cmap='gray')
                        plt.show()
                        '''
                        imgs.append([mask_class_one.astype('float32'), mask_class_two.astype('float32')])
        imgs = np.array(imgs)
        return imgs

    def save_model(self, path):
        timestamp = datetime.datetime.now()
        time = timestamp.strftime("%X").replace(':', '_')
        date = timestamp.strftime("%x").replace('/', '_')
        timestamp_str = date + ' ' + time + '.pt'
        torch.save(self.model.state_dict(), os.path.join(path, timestamp_str))

    def train(self):
        self.model.train()
        print('-----------------Begin training-----------------')
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
                batch_x = torch.unsqueeze(batch_x, dim = 1)
                
                ### Model Prediction ###
                pred_y = self.model(batch_x)

                ### Crop mask in order to match it with the prediction ###
                diff = int((batch_y.shape[2] - pred_y.shape[2])/2)
                batch_y = batch_y[:,:,diff:(batch_y.shape[2]-diff),diff:(batch_y.shape[2]-diff)]

                if verbose:
                    print('batch_x shape: ', batch_x.shape)
                    print('batch_y shape: ', batch_y.shape)           
                    print('Model predictions pred_y shape: ', pred_y.shape)

                ### Compute the training loss ###
                loss_train = self.criterion(pred_y, batch_y)
                self.train_losses.append(loss_train)

                ### Backprop and update weights ###
                loss_train.backward()
                self.optimizer.step()
            print('Epoch: {}, Loss: {}\n'.format(epoch, loss_train))

    def display_result(self):
        plt.plot(np.arange(100).astype(int)+1, self.train_losses, label='training loss')
        plt.xlabel('epoch')
        plt.ylabel('cross entroy loss')
        plt.legend()
        plt.savefig('./plot/training_loss')

def main():
    detector = Detector()
    detector.load_image('.\\dataset')
    detector.train()
    detector.save_model('.\\model')
    detector.display_result()

if __name__ == '__main__':
    main()