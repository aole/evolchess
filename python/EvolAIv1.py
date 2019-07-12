
import chess
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

bit_layers = 12 # 6 pieces for each side
learning_rate = 0.01

def create_input(board):
    posbits = []
    
    wss = chess.SquareSet()
    bss = chess.SquareSet()

    for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        wss = wss.union(board.pieces(piece, chess.WHITE))
        posbits += board.pieces(piece, chess.WHITE).tolist()

    for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        bss = bss.union(board.pieces(piece, chess.BLACK))
        posbits += board.pieces(piece, chess.BLACK).tolist()
    '''
    posbits += wss.tolist()
    posbits += bss.tolist()
    posbits += wss.union(bss).tolist()
    '''
    x = T.tensor(posbits, dtype = T.float32)
    x = x.reshape([12,8,8])
    return x

class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()
    
        kernel_size = 3
        padding = kernel_size//2
        
        self.conv_out_nodes = out_channels * 8 * 8 * 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size, padding=padding)
        self.fc1 = nn.Linear(self.conv_out_nodes, 1024)
        self.drop1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, 64*64)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(-1, self.conv_out_nodes)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        
        return x
        
class FenDataset(Dataset):
    def __init__(self, filename):
        self.all_data = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                fen, move = line[:-1].split('=')
                board = chess.Board(fen)
                x = create_input(board)
                move = chess.Move.from_uci(move)
                #y_real = T.zeros(64*64, dtype=T.long)
                pos = move.from_square*64+move.to_square
                #y_real[pos] = 1.
                self.all_data.append((x, pos))
        
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

if __name__ == '__main__':
    dataset = FenDataset('mio.txt')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = T.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    model = Model(bit_layers, 24)
    
    optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)
    max_epoch = 1000
    log_interval = 100
    
    for epoch in range(1, max_epoch+1):
        # TRAIN
        print('Training...')
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            #data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                    
        # SAVE
        T.save(model.state_dict(),"nn.pt")
        
        # TEST
        print('Testing...')
        model.eval()
        
        test_loss = 0
        correct = 0
        
        
        with T.no_grad():
            for data, labels in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, labels, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        