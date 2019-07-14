
import chess
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

bit_layers = 12 # 6 pieces for each side
learning_rate = 0.01
train = True
load_model = False

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
        
        self.conv_out_nodes = out_channels * 8 * 8 #* 2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels) # **** WOW ****
        #self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size, padding=padding)
        self.fc1 = nn.Linear(self.conv_out_nodes, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        #self.fc2 = nn.Linear(1024, 1024)
        #self.drop2 = nn.Dropout(p=0.5)
        self.fcf = nn.Linear(1024, 64*64)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        #x = F.relu(self.conv2(x))
        x = x.reshape(-1, self.conv_out_nodes)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        #x = F.relu(self.fc2(x))
        #x = self.drop2(x)
        x = self.fcf(x)
        x = F.log_softmax(x, dim=1)
        
        return x
        
class FenDataset(Dataset):
    def __init__(self, filename):
        self.all_data = []
        self.all_fen = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                fen, move = line[:-1].split('=')
                self.all_fen.append((fen, move))
                
                board = chess.Board(fen)
                x = create_input(board)
                move = chess.Move.from_uci(move)
                #y_real = T.zeros(64*64, dtype=T.long)
                pos = move.from_square*64+move.to_square
                #y_real[pos] = 1.
                self.all_data.append((x, pos, line[:-1]))
        
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

    def getFen(self, idx):
        return self.all_fen[idx]
        
if __name__ == '__main__':
    dataset = FenDataset('mio.txt')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = T.utils.data.random_split(dataset, [train_size, test_size])

    print(len(dataset), len(train_dataset), len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # NN
    model = Model(bit_layers, 24)
    # Load weights
    if load_model:
        try:
            model.load_state_dict(T.load('nn.pt'))
        except:
            pass
    
    optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)
    max_epoch = 1000
    log_interval = 200
    
    for epoch in range(1, max_epoch+1):
        # TRAIN
        if train:
            print('Training...')
            model.train()
            for batch_idx, (data, target, fen) in enumerate(train_loader):
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
            print('Loss: {:.6f}'.format(loss.item()))

            # SAVE
            T.save(model.state_dict(),"nn.pt")
        
        # TEST
        print('Testing...')
        model.eval()
        
        test_loss = 0
        correct = 0
        
        games_sample = []
        with T.no_grad():
            for data, target, fen in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                games_sample.append((fen[-1], output[-1]))
        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        # VISUAL
        line, out = games_sample[np.random.choice(len(games_sample))]
        outb = T.argmax(out).item()
        f = outb//64
        t = outb-f*64
        outmove = chess.Move(f,t)
        fen, move = line.split('=')
        board = chess.Board(fen)
        out = out.detach().numpy()
        dtype=[('move', 'S10'), ('score', float)]
        scores = []
        #out[::-1].sort()
        for m in board.legal_moves:
            pos = m.from_square*64 + m.to_square
            scores.append((board.san(m), out[pos]))
        scores = np.array(scores, dtype=dtype)
        scores[::-1].sort(order='score')
        for s in scores:
            print(s[0].decode("utf-8"),'=',round(s[1],3),end='; ')
        print()
        move = chess.Move.from_uci(move)
        print(board)
        try:
            print(board.san(move), board.san(outmove), out[outb])
        except:
            pass
        
        '''
        fen, move = test_dataset.getFen(0)
        board = chess.Board(fen)
        print(board)
        print(move)
        
        data, target = test_dataset[0]
        output = model(data)
        print('Output:', output)
        '''
        