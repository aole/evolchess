
import chess
import chess.pgn
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
        
def play_game(NN, verbose=False):
    board = chess.Board()

    while True:
        if board.is_game_over(claim_draw = True):
            result = board.result(claim_draw = True)
            if verbose:
                print(result)
            return result, board
            
        if board.turn==chess.WHITE:
            x = create_input(board)
            x.unsqueeze_(0)
            out = NN(x)[0].detach().numpy()
            chosen = None
            for i in range(10000):
                outidx = np.argmax(out)
                f = outidx//64
                t = outidx-f*64
                outmove = chess.Move(f,t)
                if outmove in board.legal_moves:
                    chosen = outmove
                    break
                out[outidx] = -10000
            if not chosen:
                print(out)
        else:
            chosen = np.random.choice(list(board.legal_moves))
            
        if verbose:
            print(board.san(chosen))
            
        board.push(chosen)
        
        if verbose:
            print(board)
    
if __name__ == '__main__':
    # NN
    model = Model(bit_layers, 24)
    # Load weights
    
    try:
        model.load_state_dict(T.load('nnStable.pt'))
    except:
        print('Error loading weights!')
        exit()

    for games in range(100):
        r, b = play_game(model)
        print('Test ('+str(games+1)+'):', r)
        game = chess.pgn.Game.from_board(b)
        filename = 'game'+str(games+1)+'.pgn'
        print(game, file=open(filename, 'w'))
    