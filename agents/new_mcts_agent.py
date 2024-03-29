# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import datetime
from collections import defaultdict
from copy import deepcopy
import random, math

#import numpy as np # REMOVE

# Code outline credit: https://ai-boson.github.io/mcts/

@register_agent("new_mcts_agent")
class NewMCTSAgent(Agent):
    def __init__(self):
        super(NewMCTSAgent, self).__init__()
        self.name = "New_MCTS_Agent"
        self.dir_map = {"u": 0, "r": 1, "d": 2, "l": 3}

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
        start_t = datetime.datetime.now()

        # Initialize MCTS Tree
        state = GameState(chess_board, 0, my_pos, adv_pos, max_step)
        root_node = MCTSNode(state)
        # p0_score, p1_score = root_node.simulate() # do first simulation
        # root_node.backpropagate(p0_score, p1_score)

        while (datetime.datetime.now() - start_t).total_seconds() < 5:
            #print("Selecting a leaf")
            #print("Tree Policy")
            selected_node = root_node.tree_policy()
            if not selected_node.state.is_endgame:
                #print("Simulate")
                p0_score, p1_score = selected_node.simulate()
            else:
                p0_score = selected_node.state.p0_score
                p1_score = selected_node.state.p1_score
            #print("Backpropagate")
            selected_node.backpropagate(p0_score, p1_score)

        if len(root_node.children) == 0:
            print("Root has no children, returning a random action")
            return state.random_action()
        else:
            most_visits = -math.inf
            for c in root_node.children:
                if c.num_visits > most_visits:
                    most_visits = c.num_visits
                    best_action = c.parent_action
            print("best action num visits: " + str(most_visits))
            return best_action

class MCTSNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.num_wins = 0
        self.num_visits = 0
        #self.fully_expanded = False
        self.untried_actions = state.get_legal_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=math.sqrt(2)):
        best_value = -math.inf
        for c in self.children:
            q_value = c.num_wins/c.num_visits + c_param * math.sqrt(math.log(self.num_visits)/c.num_visits)
            if q_value > best_value:
                best_value = q_value
                best_child = c
        #print("Best child value: " + str(best_value))
        return best_child

    def tree_policy(self):
        """
        Traverse tree using tree policy
        """
        curr_node = self
        while not curr_node.state.is_endgame: 
            # Pick an unvisited child
            if not curr_node.is_fully_expanded():
                return curr_node.expand()
            else:
                curr_node = curr_node.best_child()
        return curr_node
        # selected_node = self
        # while not selected_node.is_leaf_node(): 
        #     # Pick the best child unless we find one that hasn't been visited yet
        #     for c in self.children:
        #         if c.num_visits == 0:
        #             #print("Found unexpanded node")
        #             return c
        #     print("No unexpanded children! Select best one")
        #     best_child = selected_node.best_child()
        #     print("best child, leaf?: " + str(best_child.is_leaf_node()))
        #     input("press something")
        #     selected_node = best_child
        
        # return selected_node

    def expand(self):
        """
        Create child node for each legal action
        Return a random child node
        """   
        random_index = random.randrange(len(self.untried_actions))
        a = self.untried_actions.pop(random_index)
        next_state = self.state.move(a)
        child_node = MCTSNode(next_state, parent=self, parent_action=a)
        self.children.append(child_node)
        return child_node
        # for a in self.state.get_legal_actions():
        #     #print("Action: " + str(a))
        #     next_state = self.state.move(a)
        #     child_node = MCTSNode(next_state, parent=self, parent_action=a)
        #     self.children.append(child_node)
        # random_index = random.randrange(len(self.children))
        # return self.children[random_index]

    def simulate(self):
        curr_state = self.state  
        #input("Doing one random game iteration, Press Enter to continue...")
        while not curr_state.is_endgame:
            print("find random action")
            action = curr_state.random_action() # random default policy
            print("make move")
            curr_state = curr_state.move(action)
        #input("Done with simulation, returning scores, Press Enter to continue...")
        return curr_state.scores()

    def backpropagate(self, p0_score, p1_score):
        self.num_visits += 1
        if(self.state.player_turn == 0 and (p0_score > p1_score)):
            self.num_wins += 1
        elif(self.state.player_turn == 1 and (p1_score > p0_score)):
            self.num_wins += 1

        if self.parent:
            self.parent.backpropagate(p0_score, p1_score)

# Data structure that holds information about a game state
class GameState:
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

    # Action = (pos, dir)
    # Return list of all legal actions from state
    def get_legal_actions(self): 
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        # for each combination, check if it's a valid step; if so add it to valid actions list
        # actions = []
        # for r in range(self.board_size):
        #     for c in range(self.board_size):
        #         for dir in range(4):
        #             if(check_valid_step(self.chess_board, np.array(self.my_pos), (r, c), np.array(self.adv_pos), dir, self.max_step)):
        #                 actions.append(tuple([(r, c), dir]))

        # #random.shuffle(actions)
        # return actions

        actions = []
        # Add actions we can do from starting position
        # We can add barriers around ourselves if one does not exist in that direction yet
        cur_r, cur_c = self.my_pos
        #print("My pos: " + str(self.my_pos))
        for dir in range(4):
            if not self.chess_board[cur_r, cur_c, dir]:
                #print("Adding legal action " + str(tuple([(cur_r, cur_c), dir])))
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
                if self.chess_board[r, c, move_dir] or tuple_equal(cur_pos + move, self.adv_pos) or tuple(cur_pos + move) in visited:
                    continue

                # if not type(cur_pos) is tuple:
                #     print("cur_pos type: " + str(type(cur_pos)))
                # if not type(move) is tuple:
                #     print("move type: " + str(type(move)))
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                # if not type(next_pos) is tuple:
                #     print("next_pos type: " + str(type(next_pos)))
                #print("next_pos: " + str(next_pos))
                next_pos_r, next_pos_c = next_pos
                for barrier_dir in range(4):
                    if not self.chess_board[next_pos_r, next_pos_c, barrier_dir]:
                        #print("Adding legal action " + str(tuple([tuple(next_pos), barrier_dir])))
                        actions.append(tuple([tuple(next_pos), barrier_dir]))
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return actions

    def random_action(self): 
        new_pos = self.my_pos
        steps = random.randrange(self.max_step + 1)

        for _ in range(steps):
            r, c = self.my_pos
            dir = random.randrange(0, 4)
            m_r, m_c = self.moves[dir]
            new_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while self.chess_board[r, c, dir] or tuple_equal(new_pos, self.adv_pos):
                k += 1
                if k > 300:
                    break
                dir = random.randrange(0, 4)
                m_r, m_c = self.moves[dir]
                new_pos = (r + m_r, c + m_c)

            if k > 300:
                new_pos = self.my_pos
                break

        dir = random.randrange(4)
        r, c = new_pos
        while self.chess_board[r, c, dir]:
            dir = random.randrange(4)

        return (new_pos, dir)

    def scores(self):
        '''
        Modify according to your game or 
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        return (self.p0_score, self.p1_score)


    def move(self, action):
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

        :param action: (pos, dir)
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