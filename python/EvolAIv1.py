from __future__ import print_function

import chess
import numpy as np
import time, random

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
writer = SummaryWriter('C:/temp/tb/runs/EvolAIv1.8')

from tqdm import tqdm

#datafiles = ['smallmio.txt']
datafiles = ['mio.txt', 'bmio.txt']
train = True
load_model = False

max_epoch = 100
log_interval = 300
    
# parameters
#  1    Side to move
# 12    Position of each piece (both sides)
#  6    to squares of own pieces
#  1    to squares for oppo
#  1    to squeres for oppo king
bit_layers = 1 + \
            12 + \
             6 + \
             2
learning_rate = 0.01

# model structure
convolution_layers = 2 #3
fully_connected = 1
in_out_channel_multiplier = 2 #3

# consistent testing
T.manual_seed(4)

# check for gpu
device = T.device('cpu')
if T.cuda.is_available():
    device = T.device('cuda')
    
def create_input(board):
    posbits = chess.SquareSet(board.turn).tolist()
    
    for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        posbits += board.pieces(piece, chess.WHITE).tolist()

    for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        posbits += board.pieces(piece, chess.BLACK).tolist()
        
    
    # all attack squares
    to_queen_sqs = chess.SquareSet()
    to_rook_sqs = chess.SquareSet()
    to_bishop_sqs = chess.SquareSet()
    to_knight_sqs = chess.SquareSet()
    to_king_sqs = chess.SquareSet()
    to_pawn_sqs = chess.SquareSet()
    for move in board.legal_moves:
        if board.san(move)[0]=='Q':
            to_queen_sqs.add(move.to_square)
        elif board.san(move)[0]=='R':
            to_rook_sqs.add(move.to_square)
        elif board.san(move)[0]=='B':
            to_bishop_sqs.add(move.to_square)
        elif board.san(move)[0]=='N':
            to_knight_sqs.add(move.to_square)
        elif board.san(move)[0]=='K' or board.san(move)[0]=='O':
            to_king_sqs.add(move.to_square)
        else:
            to_pawn_sqs.add(move.to_square)
    posbits += to_queen_sqs.tolist()+to_rook_sqs.tolist()+to_bishop_sqs.tolist()+to_knight_sqs.tolist()+to_king_sqs.tolist()+to_pawn_sqs.tolist()
    
    # all opponent attack squares
    to_sqs = chess.SquareSet()
    to_king_sqs = chess.SquareSet()
    board.turn = not board.turn
    for move in board.legal_moves:
        to_sqs.add(move.to_square)
        if board.san(move)[0]=='K' or board.san(move)[0]=='O':
            to_king_sqs.add(move.to_square)
    board.turn = not board.turn
    posbits += to_sqs.tolist()
    posbits += to_king_sqs.tolist()
    
    #en passant square
    #posbits += (chess.SquareSet(chess.BB_SQUARES[board.ep_square]) if board.ep_square else chess.SquareSet()).tolist()
    
    x = T.tensor(posbits, dtype = T.float32)
    x = x.reshape([bit_layers,8,8])
    return x

class FenDataset(Dataset):
    def __init__(self, filenames):
        self.all_data = []
        self.all_fen = []
        print('Files to load:',len(filenames))
        for filename in filenames:
            with open(filename, 'r') as f:
                lines = f.readlines()
                tot = len(lines)
                print(filename,'Total:', tot)
                pbar = tqdm(total=tot)
                for i, line in enumerate(lines):
                    fen, move = line[:-1].split(',')
                    self.all_fen.append((fen, move))
                    
                    board = chess.Board(fen)
                    x = create_input(board)
                    move = chess.Move.from_uci(move)
                    pos = move.from_square*64+move.to_square
                    self.all_data.append((x, pos, line[:-1]))
                    pbar.update()
                pbar.close()
                
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

    def getFen(self, idx):
        return self.all_fen[idx]
        
class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
    
        kernel_size = 3
        padding = kernel_size//2
        
        out_channels = in_channels * in_out_channel_multiplier
        self.conv_out_nodes = out_channels * 8 * 8
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels) # **** WOW ****
        
        if convolution_layers>=2:
            self.conv2 = nn.Conv2d(out_channels, out_channels*2, kernel_size, padding=padding)
            self.bn2 = nn.BatchNorm2d(out_channels*2) # **** WOW ****
            self.conv_out_nodes *= 2

        if convolution_layers>=3:
            self.conv3 = nn.Conv2d(out_channels*2, out_channels*4, kernel_size, padding=padding)
            self.bn3 = nn.BatchNorm2d(out_channels*4)
            self.conv_out_nodes *= 2
        
        self.fc1 = nn.Linear(self.conv_out_nodes, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        
        if fully_connected>=2:
            self.fc2 = nn.Linear(1024, 1024)
            self.drop2 = nn.Dropout(p=0.5)
        
        self.fcf = nn.Linear(1024, 64*64)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        
        if convolution_layers >= 2:
            x = F.relu(self.conv2(x))
            x = self.bn2(x)
            
        if convolution_layers >= 3:
            x = F.relu(self.conv3(x))
            x = self.bn3(x)
        
        x = x.view(-1, self.conv_out_nodes)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        
        if fully_connected >= 2:
            x = F.relu(self.fc2(x))
            x = self.drop2(x)
            
        x = self.fcf(x)
        x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1) -- bad
        
        return x
        
if __name__ == '__main__':
    print('Setting up database')
    dataset = FenDataset(datafiles)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = T.utils.data.random_split(dataset, [train_size, test_size])

    print(len(dataset), len(train_dataset), len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # same some test instances
    testfile = open('evaluate.txt', 'w')
    
    # NN
    model = Model(bit_layers)
    model.to(device)
    
    # Load weights
    if load_model:
        try:
            model.load_state_dict(T.load('nn.pt'))
        except:
            pass
    
    optimizer = T.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    #optimizer = T.optim.Adam(model.parameters(), lr=learning_rate) # not learning
    
    for epoch in range(max_epoch):
        # TRAIN
        if train:
            print('Training...')
            t0 = time.time()
            model.train()
            for batch_idx, (data, target, fen) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                        
            writer.add_scalar('Train Loss', loss.item(), epoch)
            writer.flush()
            
            timediff = time.time()-t0
            mins = timediff//60
            secs = timediff-mins*60
            print('Time taken:', str(mins)+':'+str(int(secs)))
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
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                rndidx = np.random.choice(len(fen))
                games_sample.append((fen[rndidx], output[rndidx]))
        test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct / len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))
        
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Accuracy', accuracy, epoch)
        writer.flush()
        
        # VISUAL
        print('Write evaluation...')
        testfile.write('Epoch: '+str(epoch)+'\n')
        games_sample = random.sample(games_sample, 10)
        for idx, sample in enumerate(games_sample):
            line, out = sample
            outb = T.argmax(out).item()
            f = outb//64
            t = outb-f*64
            outmove = chess.Move(f,t)
            fen, move = line.split(',')
            board = chess.Board(fen)
            out = out.cpu().detach().numpy()
            
            move = chess.Move.from_uci(move)
            testfile.write(line+'\n'+str(board)+'\n')
            testfile.write('White' if board.turn else 'Black')
            testfile.write(' = '+str(outmove)+'\n')
            try:
                testfile.write('Best:', board.san(move), 'AI:', board.san(outmove), out[outb])
            except:
                pass
            
            dtype=[('move', 'S10'), ('score', float)]
            scores = []
            #out[::-1].sort()
            for m in board.legal_moves:
                pos = m.from_square*64 + m.to_square
                scores.append((board.san(m), out[pos]))
            scores = np.array(scores, dtype=dtype)
            scores[::-1].sort(order='score')
            for s in scores:
                m = s[0].decode("utf-8")
                if m[-1]=='#':
                    m = '©   '+m+'   ®'
                msg = str(m)+'='+str(round(s[1],3))+'; '
                testfile.write(msg)
            testfile.write('\n==============\n')
        
        testfile.write('\n\n')
        testfile.flush()
        print()
    
    testfile.close()
    