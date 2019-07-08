from __future__ import print_function

import os, glob

import numpy as np
import chess
import chess.pgn

import torch
import torch.nn as nn
import torch.functional as F

def create_input(board):
    posbits = []
    for side in [chess.WHITE, chess.BLACK]:
        for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            posbits += board.pieces(piece, side).tolist()
            
    x = torch.as_tensor(posbits, dtype = torch.float32)
    return torch.unsqueeze(x, 0)
        
class ChessAINew2(nn.Module):
    def __init__(self):
        super(ChessAINew2, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(64*6*2, 1024),
            nn.ReLU(),
            nn.Linear(1024,1),
            nn.Tanh())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.last_pred = 0

            
    def best_score(self, board, depth=0):
        if board.is_game_over(claim_draw = True):
            result = board.result(claim_draw = True)
            return self.model(create_input(board))
        
        best = [-10.]
        for move in board.legal_moves:
            board.push(move)
            
            if depth==0:
                out = self.model(create_input(board))
            else:
                out = self.best_score(board, depth-1)
                
            board.pop()
            
            if out[0]>best[0]:
                best = out
            
        return best
    
    def find(self, board, explore):
        scores = []
        scores_f = []
        
        for move in board.legal_moves:
            board.push(move)
            out = self.best_score(board)
            board.pop()
            scores.append([out, move])
            scores_f.append(out.detach().numpy()[0][0]+1)
        
        #print(scores_f)
        p = np.array(scores_f)
        p /= p.sum()
        
        if explore:
            best_index = np.random.choice(len(scores), p=p)
        else:
            best_index = np.random.choice(len(scores))
        self.last_pred = scores[best_index][0]
        return scores[best_index][1]
    
    def learn(self, y):
        print(self.last_pred, y)
        loss = self.criterion(self.last_pred, torch.as_tensor([y], dtype = torch.float32))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class ChessAINew(nn.Module):
    def __init__(self):
        super(ChessAINew, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(64*6*2, 1024),
            nn.ReLU(),
            nn.Linear(1024,1),
            nn.Tanh())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.last_pred = 0

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features
        
    def find(self, board, explore):
        scores = []
        scores_f = []
        
        for move in board.legal_moves:
            board.push(move)
            
            out = self.model(create_input(board))
                    
            board.pop()
            
            scores.append([out, move])
            scores_f.append(out.detach().numpy()[0][0]+1)
            
        p = np.array(scores_f)
        p /= p.sum()
        
        if explore:
            best_index = np.random.choice(len(scores), p=p)
        else:
            best_index = np.random.choice(len(scores))
            
        self.last_pred = scores[best_index][0]
        return scores[best_index][1]
        
    def learn(self, y):
        loss = self.criterion(self.last_pred, torch.as_tensor([y], dtype = torch.float32))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
class ChessAI:
    def find(self, board, explore):
        moves = board.legal_moves
        chosen = np.random.choice(list(moves))
        return chosen
    
    def learn(self, y):
        pass
        
# self play
def play_game(NN1, NN2, verbose=False, explore=False):
    board = chess.Board()

    while True:
        if board.is_game_over(claim_draw = True):
            result = board.result(claim_draw = True)
            if verbose:
                print(result)
            return result, board
            
        if board.turn==chess.WHITE:
            chosen = NN1.find(board, explore)
        else:
            chosen = NN2.find(board, explore)
            
        if verbose:
            print(board.san(chosen))
            
        board.push(chosen)
        
        if verbose:
            print(board)
    
ai = ChessAI()
ain = ChessAINew2()

wins = 0
losses = 0
draws = 0

fileList = glob.glob('game*.pgn')
print('deleting',len(fileList),'files.')
for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        pass
        
total_games = 10000
test_every = 10

print('Running games')
for games in range(total_games):
    result, board = play_game(ain, ai, explore=True)
    
    if result=='1-0':
        wins += 1
        ain.learn(1)
        ai.learn(-1)
    elif result=='0-1':
        losses += 1
        ai.learn(1)
        ain.learn(-1)
    elif result=='1/2-1/2':
        draws += 1
        ai.learn(0)
        ain.learn(0)
    print((games+1), result, ':', wins, draws, losses)

    if (games+1)%test_every==0:
        result, board = play_game(ain, ai)
        print('Test:', result)
        game = chess.pgn.Game.from_board(board)
        filename = 'game'+str(games+1)+'.pgn'
        print(game, file=open(filename, 'w'))
        
assert(total_games==wins+losses+draws)

print('Wins:',wins)
print('Losses:',losses)
print('Draws:',draws)
