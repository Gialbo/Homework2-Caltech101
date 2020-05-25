from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # filename of the splits
        self.paths = [] # list containing the paths to the images
        self.labels = [] # list containing the labels of the images encoded as integers
        self.labels_str = [] # list containing the labels of the images as strings

        # Stores the number of assigned labels so far
        label_code_counter = 0
        
        with open (os.path.join("Caltech101", split+'.txt'), 'r') as f:
            for line in f:
                # We want to discard the BACKGROUND class
                if not line.startswith("BACKGROUND"):
                    # Getting the label as string
                    label = line.split("/")[0]

                    # if the label is not inserted yet...
                    if label not in self.labels_str:
                        # assign a numerical code to the label and insert in path
                        self.paths.append(line.rstrip("\n"))
                        self.labels.append(label_code_counter)

                        # updating labels with new one
                        label_code_counter += 1
                        self.labels_str.append(label)
                    # if the label is already present in the dataset...
                    else:
                        #fetch it's numerical code
                        label_code = self.labels_str.index(label)
                        # insert entry in the informations dataset
                        self.paths.append(line.rstrip("\n"))
                        self.labels.append(label_code)





    def __getitem__(self, index):
        '''
        __getitem__ access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        # Loading image
        image = pil_loader(os.path.join(self.root, self.paths[index]))

        # Gettin labels
        label = self.labels[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.labels)
        return length