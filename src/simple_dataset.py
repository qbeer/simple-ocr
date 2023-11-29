import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image
import pandas as pd

class OCRDataset(Dataset):
    def __init__(self, subset=2,
                 base_path='./data',
                 csv='./data/train.csv',
                 chars = None):
        df = pd.read_csv(csv)
        df['text'] = df['text'].apply(lambda x: x.replace('str', ''))
        self.df = (df[df['subset'] == subset])
        
        self.image_paths = [ Path(os.path.join(base_path, file_name)) for file_name in self.df['file_path'].values ]
        if chars is None:
            self.chars = sorted(list(set(''.join([ str(val) for val in self.df['text'].values]))))
            self.transforms=T.Compose([
                     T.Resize(size=(64, 128)),
                     T.ToTensor()
                 ])
        else:
            self.chars = chars
            self.transforms=T.Compose([
                     T.Resize(size=(64, 128)),
                     T.ToTensor()
                 ])
        self.subset = subset
        
        self.idx2char = { idx: char for idx, char in enumerate(self.chars) }
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
            """
            Retrieves the image and corresponding text encoding at the given index.

            Args:
                idx (int): The index of the sample to retrieve.

            Returns:
                tuple: A tuple containing the image and text encoding.
            """
            path = self.image_paths[idx]
            img = Image.open(path)
            img = self.transforms(img)
            text = self.df['text'].values[idx]
            
            # NOTE: Create a one-hot encoding for the text
            #       the encoding is of size (subset, len(chars))
            text_encoding = torch.zeros(size=(self.subset, len(self.chars)))
            for sample_ind, char in enumerate(text):
                char_ind = self.chars.index(char)
                text_encoding[sample_ind, char_ind] += 1.
            
            return img, text_encoding
