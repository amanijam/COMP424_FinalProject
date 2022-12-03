# Student agent: Add your own agent here
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import sys
import datetime

import numpy as np
from collections import defaultdict


@register_agent("mctslight_agent")
class MCTSLightAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(MCTSLightAgent, self).__init__()
        self.name = "MCTSLightAgent"
        self.dir_map = {"u": 0, "r": 1, "d": 2, "l": 3}

        # root_state: Game simulator that helps us undersyand the same simulation 
        # root (Node) : root of the tree search
        # run_time (int): tine per run
        # node_count (int): whole nodes in tree
        # num_rollouts (int): num of rollouts for each search

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
        # create new tree
        # self.root_node = MontreCarloTreeSearchNode(state)
        # start_time = datetime.datetime.now()
        # while (datetime.datetime.now() - start_time).total_seconds < 1.9:
        #   SELECTION
        #   node = select()

        #   EXPANSION
        #   if node.pending_moves != set():
        #       node = expand(node)

        #   SIMULATION
        #   result = simlate(node.state)

        #   UPDATE
        #   update(node, result)

        # Return most promision child
        # return max(self.root_node.children, key=lambda n: n.value).action

        # dummy return
        return my_pos, self.dir_map["u"]


class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        # Action
        # Value
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(
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

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves): 
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 100
        for i in range(simulation_no):  
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.)

    # Data structure that holds information about a game state
    class GameState:
        def __init__(self, chess_board, player_turn, p_pos, adv_pos, max_step):
            self.chess_board = chess_board 
            self.board_size = chess_board.shape[0]
            self.player_turn = player_turn
            self.p_pos = p_pos
            self.adv_pos = adv_pos
            self.max_step = max_step

             # Moves (Up, Right, Down, Left)
            self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))


        # Action = (pos, dir)
        def get_legal_actions(self): 
            '''
            Modify according to your game or
            needs. Constructs a list of all
            possible actions from current state.
            Returns a list.
            '''
            # for each combination, check if it's a valid step; if so add it to valid actions list
            actions = []
            for r in range(self.board_size):
                for c in range(self.board_size):
                    for dir in range(0):
                        if(self.check_valid_step(self.my_pos, (r, c), dir)):
                            actions.append((r, c), dir)


        def is_game_over(self):
            '''
            Modify according to your game or 
            needs. It is the game over condition
            and depends on your game. Returns
            true or false
            '''

        def game_result(self):
            '''
            Modify according to your game or 
            needs. Returns 1 or 0 or -1 depending
            on your state corresponding to win,
            tie or a loss.
            '''

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
            next_pos, dir = action
            r, c = next_pos
            self.set_barrier(r, c, dir)

        def set_barrier(self, r, c, dir):
            # Set the barrier to True
            self.chess_board[r, c, dir] = True
            # Set the opposite barrier to True
            move = self.moves[dir]
            self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

        def check_endgame(self):
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
            # Union-Find
            father = dict()
            for r in range(self.board_size):
                for c in range(self.board_size):
                    father[(r, c)] = (r, c)

            def find(pos):
                if father[pos] != pos:
                    father[pos] = find(father[pos])
                return father[pos]

            def union(pos1, pos2):
                father[pos1] = pos2

            for r in range(self.board_size):
                for c in range(self.board_size):
                    for dir, move in enumerate(
                        self.moves[1:3]
                    ):  # Only check down and right
                        if self.chess_board[r, c, dir + 1]:
                            continue
                        pos_a = find((r, c))
                        pos_b = find((r + move[0], c + move[1]))
                        if pos_a != pos_b:
                            union(pos_a, pos_b)

            for r in range(self.board_size):
                for c in range(self.board_size):
                    find((r, c))

            my_r = find(tuple(self.my_pos))
            adv_r = find(tuple(self.adv_pos))
            my_score = list(father.values()).count(my_r)
            adv_score = list(father.values()).count(adv_r)
            if my_r == adv_r:
                return False, my_score, adv_score

            return True, my_score, adv_score

        def check_valid_step(self, start_pos, end_pos, barrier_dir):
            """
            Check if the step the agent takes is valid (reachable and within max steps).

            Parameters
            ----------
            start_pos : tuple
                The start position of the agent.
            end_pos : np.ndarray
                The end position of the agent.
            barrier_dir : int
                The direction of the barrier.
            """
            # Endpoint already has barrier or is boarder
            r, c = end_pos
            if self.chess_board[r, c, barrier_dir]:
                return False
            if np.array_equal(start_pos, end_pos):
                return True

            # Get position of the adversary
            adv_pos = self.adv_pos

            # BFS
            state_queue = [(start_pos, 0)]
            visited = {tuple(start_pos)}
            is_reached = False
            while state_queue and not is_reached:
                cur_pos, cur_step = state_queue.pop(0)
                r, c = cur_pos
                if cur_step == self.max_step:
                    break
                for dir, move in enumerate(self.moves):
                    if self.chess_board[r, c, dir]:
                        continue
                    
                    next_pos = cur_pos + move
                    if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                        continue
                    if np.array_equal(next_pos, end_pos):
                        is_reached = True
                        break

                    visited.add(tuple(next_pos))
                    state_queue.append((next_pos, cur_step + 1))

            return is_reached

def main():

    root = MonteCarloTreeSearchNode(state = initial_state)
    selected_node = root.best_action()
    return 
