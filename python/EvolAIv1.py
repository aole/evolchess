from __future__ import print_function

import chess
import numpy as np
import time, random

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
writer = SummaryWriter('C:/temp/tb/runs/EvolAIv1.9')

from tqdm import tqdm

# consistent testing
T.manual_seed(4)

datafiles = ['smallmio.txt']
#datafiles = ['mio.txt']
#datafiles = ['mio.txt', 'bmio.txt']

train = True
load_model = False
save_model = True

max_epoch = 500
log_interval = 300
early_stopping_patience = 10
best_score = None
best_accuracy = None

# parameters
#  1    Side to move
# 12    Position of each piece (both sides)
# 12    to squares of each pieces (both sides)
bit_layers = 1 + \
            12 + \
            12
learning_rate = 0.01

# model structure
convolution_layers = 2 #3
fully_connected = 1
in_out_channel_multiplier = 2 #3

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
    to_sqs = [chess.SquareSet() for x in range(7)]
    for i, p in board.piece_map().items():
        for t in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING, chess.PAWN]:
            if p.piece_type==t and p.color==board.turn:
                to_sqs[p.piece_type] = to_sqs[p.piece_type].union(board.attacks(i))
            
    posbits += to_sqs[1].tolist()+to_sqs[2].tolist()+to_sqs[3].tolist()+to_sqs[4].tolist()+to_sqs[5].tolist()+to_sqs[6].tolist()
    
    # all opponent attack squares
    board.turn = not board.turn
    to_sqs = [chess.SquareSet() for x in range(7)]
    for i, p in board.piece_map().items():
        for t in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING, chess.PAWN]:
            if p.color==board.turn:
                to_sqs[p.piece_type] = to_sqs[p.piece_type].union(board.attacks(i))
            
    posbits += to_sqs[1].tolist()+to_sqs[2].tolist()+to_sqs[3].tolist()+to_sqs[4].tolist()+to_sqs[5].tolist()+to_sqs[6].tolist()
    board.turn = not board.turn
    
    #en passant square
    #posbits += (chess.SquareSet(chess.BB_SQUARES[board.ep_square]) if board.ep_square else chess.SquareSet()).tolist()
    
    x = T.tensor(posbits, dtype = T.float32)
    x = x.reshape([bit_layers,8,8])
    return x

class FenDataset(Dataset):
    def __init__(self, filenames):
        self.all_data = []
        self.all_fen = []
        for idx, filename in enumerate(filenames):
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc='Loading '+filename+' ('+str(idx+1)+'/'+str(len(filenames))+')'):
                    fen, move = line[:-1].split(',')
                    self.all_fen.append((fen, move))
                    
                    board = chess.Board(fen)
                    x = create_input(board)
                    move = chess.Move.from_uci(move)
                    pos = move.from_square*64+move.to_square
                    self.all_data.append((x, pos, line[:-1]))
                
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
        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        
        if convolution_layers >= 2:
            x = F.leaky_relu(self.conv2(x))
            x = self.bn2(x)
            
        if convolution_layers >= 3:
            x = F.leaky_relu(self.conv3(x))
            x = self.bn3(x)
        
        x = x.view(-1, self.conv_out_nodes)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop1(x)
        
        if fully_connected >= 2:
            x = F.leaky_relu(self.fc2(x))
            x = self.drop2(x)
            
        #x = self.fcf(x)
        x = F.leaky_relu(self.fcf(x))
        
        x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1) -- bad
        
        return x
        
if __name__ == '__main__':
    dataset = FenDataset(datafiles)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = T.utils.data.random_split(dataset, [train_size, test_size])

    print('Samples:',len(dataset), 'Total,', len(train_dataset),'Train,', len(test_dataset),'Test.')
    
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
            print('Loading model!')
            model.load_state_dict(T.load('nn.pt'))
        except:
            print('Error in loading model')
    
    optimizer = T.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    #optimizer = T.optim.Adam(model.parameters(), lr=learning_rate) # not learning
    
    patience = 0
    for epoch in range(max_epoch):
        # TRAIN
        if train:
            model.train()
            pbar = tqdm(total=len(train_loader))
            pbar.set_description('Training ('+str(epoch)+')')
            for batch_idx, (data, target, fen) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % log_interval == 0:
                    pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            pbar.close()
            
            writer.add_scalar('Train Loss', loss.item(), epoch)
            writer.flush()
            
        # TEST
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
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))
        
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Accuracy', accuracy, epoch)
        writer.flush()
        
        # VISUAL
        #print('Writing evaluation...')
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
    
        score = -test_loss
        # early stopping
        if not best_score or score>best_score or correct>best_accuracy:
            best_score = score
            best_accuracy = correct
            patience = 0
            # SAVE
            if save_model:
                print('Saving Model!')
                T.save(model.state_dict(),"nn.pt")
        else:
            patience += 1
            if patience > early_stopping_patience:
                print('Stopping early!!!')
                break
            
        print()
        
    testfile.close()
    