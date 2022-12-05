# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import datetime
from collections import defaultdict
from copy import deepcopy
import random, math

import numpy as np # REMOVE

# Code outline credit: https://ai-boson.github.io/mcts/

@register_agent("mcts_9x9_tto_agent")
class MCTS_9x9_TTOAgent(Agent):
    def __init__(self):
        super(MCTS_9x9_TTOAgent, self).__init__()
        self.name = "MCTS_9x9_TTO_Agent"
        self.dir_map = {"u": 0, "r": 1, "d": 2, "l": 3}
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        start_time = datetime.datetime.now()
        state = GameState(chess_board, 0, my_pos, adv_pos, max_step)
        root_node = MCTSNode(state)
        #simulation_no = 100

        while (datetime.datetime.now() - start_time).total_seconds() < 1.95:
            #print("Tree Policy")
            v = root_node.tree_policy()
            p0_score, p1_score = v.rollout() # return p0_score, p1_score
            v.backpropagate(p0_score, p1_score)

        most_visits = -math.inf
        for c in root_node.children:
            if c.num_visits > most_visits:
                most_visits = c.num_visits
                best_action = c.parent_action
        #print("best action num visits: " + str(most_visits))
        return best_action

class MCTSNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.num_visits = 0
        self.results = defaultdict(int)
        self.results[1] = 0
        self.results[-1] = 0
        self.untried_actions = self.state.get_legal_actions()

    def q(self):
        wins = self.results[1]
        loses = self.results[-1]
        return wins - loses

    def n(self):
        return self.num_visits

    def expand(self):
        random_index = random.randrange(len(self.untried_actions))
        action = self.untried_actions.pop(random_index)
        next_state = self.state.move(action)
        child_node = MCTSNode(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node 

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)

        return current_rollout_state.game_result()

    def backpropagate(self, p0_score, p1_score):
        self.num_visits += 1
        if self.state.player_turn == 0:
            if p0_score > p1_score:
                self.results[1] += 1
            elif p0_score < p1_score:
                self.results[-1] += 1
            else:
                self.results[0] += 1
        elif self.state.player_turn == 1:
            if p1_score > p0_score:
                self.results[1] += 1
            elif p1_score < p0_score:
                self.results[-1] += 1
            else:
                self.results[0] += 1

        if self.parent:
            self.parent.backpropagate(p0_score, p1_score)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def select_child(self, c_param=math.sqrt(2)): 
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[random.randrange(len(possible_moves))]


    def tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.select_child()
        return current_node

class GameState():
    def __init__(self, chess_board, player_turn, p0_pos, p1_pos, max_step):
        self.chess_board = chess_board 
        self.board_size = chess_board.shape[0]

        self.player_turn = player_turn # 0 or 1 for p0 or p1

        if player_turn == 0:
            self.my_pos = p0_pos
            self.adv_pos = p1_pos
        else:
            self.my_pos = p1_pos
            self.adv_pos = p0_pos

        self.max_step = max_step

        self.is_endgame, self.p0_score, self.p1_score = check_endgame(chess_board, p0_pos, p1_pos)

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        self.legal_actions = self.generate_legal_actions()

    # Action = (pos, dir)
    # Return list of all legal actions from state
    def generate_legal_actions(self): 
        actions = []
        # Add actions we can do from starting position
        # We can add barriers around ourselves if one does not exist in that direction yet
        cur_r, cur_c = self.my_pos

        for dir in range(4):
            if not self.chess_board[cur_r, cur_c, dir]:
                actions.append(tuple([(cur_r, cur_c), dir]))

        # BFS
        state_queue = [(self.my_pos, 0)]
        visited = {self.my_pos}
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for move_dir, move in enumerate(self.moves):
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if self.chess_board[r, c, move_dir] or tuple_equal(next_pos, self.adv_pos) or next_pos in visited:
                    continue
                next_pos_r, next_pos_c = next_pos
                for barrier_dir in range(4):
                    if not self.chess_board[next_pos_r, next_pos_c, barrier_dir]:
                        actions.append(tuple([tuple(next_pos), barrier_dir]))
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return actions

    def get_legal_actions(self): 
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        return self.legal_actions

    def is_game_over(self):
        '''
        Modify according to your game or 
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        return self.is_endgame or len(self.legal_actions) == 0
    
    def game_result(self):
        '''
        Modify according to your game or 
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        return (self.p0_score, self.p1_score)

    def move(self,action):
        '''
        Modify according to your game or 
        needs. Changes the state of your 
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board 
        position is empty. If you place x in
        row 2 column 3, then it would be some 
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns 
        the new state after making a move.
        '''
        # Returns a new GameState
        next_pos, dir = action

        # Get players' next positions
        if self.player_turn == 0:
            p0_pos = next_pos
            p1_pos = self.adv_pos
        else:
            p1_pos = next_pos
            p0_pos = self.adv_pos

        # Update chess_board (Set the barrier to True)
        r, c = next_pos
        # print("next_pos: " + str(next_pos))
        updated_chessboard = set_barrier(self.chess_board, r, c, dir)
        
        # Switch player turn
        next_player_turn = 1 - self.player_turn

        next_state = GameState(updated_chessboard, next_player_turn, p0_pos, p1_pos, self.max_step)
        return next_state

def tuple_equal(t1: tuple, t2: tuple):
    return (t1[0] == t2[0] and t1[1] == t2[1])

def set_barrier(old_chessboard, r, c, dir):
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    new_chessboard = deepcopy(old_chessboard)

    # Set the barrier to True
    new_chessboard[r, c, dir] = True

    # Set the opposite barrier to True
    move = moves[dir]

    new_chessboard[r + move[0], c + move[1], opposites[dir]] = True

    return new_chessboard

def check_endgame(chess_board, p0_pos, p1_pos):
    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """
    board_size = chess_board.shape[0]
    
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    # Union-Find
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                moves[1:3]
            ):  # Only check down and right
                if chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))
    p0_r = find(tuple(p0_pos))
    p1_r = find(tuple(p1_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score, p1_score
    
    return True, p0_score, p1_score
