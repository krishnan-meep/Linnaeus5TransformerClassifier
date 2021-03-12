import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, k, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, k)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, k, 2).float()) * (-np.log(10000.0)/k)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, k, n_classes):
        super(Transformer, self).__init__()
        self.k = k
        
        self.pos_encoder = PositionalEncoding(k)
        encoder_layers = nn.TransformerEncoderLayer(k, 8, k*2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)
        self.decoder = nn.Linear(k, n_classes)

        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x * np.sqrt(self.k)
        #x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim = 1)
        x = self.decoder(x)
        return x

class PatcherDataset(Dataset):
    def __init__(self, root_dir, sliding_window_size = (16, 16)):
        self.root_dir = root_dir
        self.sliding_window_size = sliding_window_size
        self.image_list = []
        self.target = []

        for i, folder in enumerate(os.listdir(self.root_dir)):
            for file in os.listdir(os.path.join(self.root_dir, folder)):
                self.image_list.append(os.path.join(self.root_dir, folder, file))
                self.target.append(i)

    def __len__(self):
        return len(self.image_list)

    def get_no_of_classes(self):
        return len(set(self.target))

    def __getitem__(self, idx):
        img = cv2.imread(self.image_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255

        img_patches = []
        for i in range(0, img.shape[0], self.sliding_window_size[0]):
            for j in range(0, img.shape[1], self.sliding_window_size[1]):
                patch = img[i:i+self.sliding_window_size[0], j:j+self.sliding_window_size[1]]
                patch = patch.ravel()
                img_patches.append(patch)

        img_patches = torch.Tensor(img_patches)
        target = torch.LongTensor([self.target[idx]])
        return img_patches, target


if __name__ == '__main__':
    dataset = PatcherDataset("../PreTrained Model Studies/Linnaeus 5 64x64/train")
    test_dataset = PatcherDataset("../PreTrained Model Studies/Linnaeus 5 64x64/test")
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = Transformer(768, dataset.get_no_of_classes())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1.0)

    epochs = 1
    for e in range(epochs):
        for i, (data, label) in enumerate(iter(dataset_loader)):
            optimizer.zero_grad()

            out = model(data)
            L = criterion(out, label.squeeze(1))

            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
            if i%10 == 0:
                print("{}/{} Loss : {}".format(i, len(dataset_loader), L))

        acc_list= []
        for data, label in tqdm(iter(test_dataset_loader)):
            out = model(data)
            pred = F.log_softmax(out, dim = 1).argmax(dim = 1)
            correct = (pred == label.squeeze(1)).float().sum()
            acc = correct/data.shape[0]
            acc_list.append(acc)
        acc = round(np.array(acc_list).mean(), 2)

        print("Testing Acc: {}".format(acc))

                
