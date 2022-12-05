# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from time import time
from copy import deepcopy
import random, math

# Code outline credit: https://github.com/masouduut94/MCTS-agent-python.git  

@register_agent("mcts_ucb1_tuned_agent")
class MCTS_UCB1_TunedAgent(Agent):
    def __init__(self):
        super(MCTS_UCB1_TunedAgent, self).__init__()
        self.name = "MCTS_UB1_Tuned_Agent"
        self.autoplay = True

        self.root_state = None
        self.root = None

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
        """
        start_time = time()
        buffer_time = 0.01 + 0.005*(chess_board.shape[0] - 5)

        # Search
        if self.root is None:
            # It is our first turn, we get 30 to search the tree
            init_state = GameState(chess_board, my_pos, adv_pos, max_step)
            self.set_gamestate(init_state)
            self.search(start_time + 6 - time() - buffer_time)
        else:
            # It is not our first turn
            # Generate out adversary's last action to move forward in the tree
            adv_r, adv_c = adv_pos
            last_board_state = self.root_state.chess_board
            past_borders = last_board_state[adv_r, adv_c]
            for d in range(4):
                if past_borders[d] != chess_board[adv_r, adv_c, d]:
                    adv_dir = d
            self.take_action((adv_pos, adv_dir))
            self.search(start_time + 2 - time() - buffer_time)
        
        best_action = self.best_action()
        self.take_action(best_action)

        return best_action

    def search(self, time_budget) -> None:
        """
        Search and update the search tree for a
        specified amount of time in seconds.
        """
        start_time = time()

        # do until we exceed our time budget
        while time() - start_time < time_budget:
            node, state = self.select_node()
            turn = state.turn()
            outcome = self.roll_out(state) 
            self.backup(node, turn, outcome)

    def select_node(self) -> tuple:
        """
        Select a node in the tree to preform a single simulation from.
        """
        node = self.root
        state = deepcopy(self.root_state)

        # stop if we find reach a leaf node
        while len(node.children) != 0:
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values()
                         if n.value == max_value]
            node = random.choice(max_nodes)
            state.play(node.action)

            # if some child node has not been explored select it before expanding
            # other children
            if node.N == 0:
                return node, state

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.play(node.action)
        return node, state
    
    @staticmethod
    def expand(parent, state) -> bool:
        """
        Generate the children of the passed "parent" node based on the available
        actions in the passed gamestate and add them to the tree.

        Returns:
            bool: returns false If node is leaf (the game has ended).
        """
        children = []
        if state.winner != None:
            # game is over at this node so nothing to expand
            return False

        for action in state.actions():
            children.append(UCB1Tuned_MCTSNode(action, parent))

        parent.add_children(children)
        return True

    @staticmethod
    def roll_out(state) -> int:
        """
        Simulate an entirely random game from the passed state and return the winning
        player.

        Args:
            state: game state

        Returns:
            int: winner of the game
        """
        actions = state.actions()  # Get a list of all possible moves in current state of the game

        while state.winner == None:
            move = random.choice(actions)
            state.play(move)
            actions.remove(move)

        return state.winner
    
    @staticmethod
    def backup(node, turn: int, outcome: int) -> None:
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.

        Args:
            node:
            turn: winner turn
            outcome: outcome of the rollout
        """
        # Careful: The reward is calculated for player who just played
        # at the node and not the next player to play
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            reward = 0 if reward == 1 else 1

    def best_action(self) -> tuple:
        """
        Return the best action according to the current tree.
        Returns:
            best action in terms of the most simulations number unless the game is over
        """
        # Should never happen
        if self.root_state.winner != None:
            print("Game already over...?")
            return -1

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        bestchild = random.choice(max_nodes)
        return bestchild.action

    def take_action(self, action: tuple) -> None:
        """
        Make the passed action and update the tree appropriately. It is
        designed to let the player choose an action manually (which might
        not be the best action).
        Args:
            action:
        """
        if action in self.root.children:
            child = self.root.children[action]
            child.parent = None
            self.root = child
            self.root_state.play(child.action)
            return

        # if for whatever reason the action is not in the children of
        # the root just throw out the tree and start over
        self.root_state.play(action)
        self.root = UCB1Tuned_MCTSNode()

    def set_gamestate(self, state) -> None:
        """
        Set the root_state of the tree to the passed gamestate, this clears all
        the information stored in the tree since none of it applies to the new
        state.
        """
        self.root_state = deepcopy(state)
        self.root = UCB1Tuned_MCTSNode()

            
class UCB1Tuned_MCTSNode:
    """
    Node for the MCTS. 
    Stores the action applied to reach this node from its parent,
    stats for the associated game position, children, parent and outcome
    (outcome==None unless the position ends the game).
    Args:
        action:
        parent:
        N (int): times this position was visited
        Q (int): average reward (wins-losses) from this position
        children (dict): dictionary of successive nodes
        outcome (int): If node is a leaf, then outcome indicates
                       the winner, else None
    """

    def __init__(self, action: tuple = None, parent: object = None):
        self.action = action
        self.parent = parent
        self.N = 0 
        self.Q = 0  
        self.children = {}
        self.outcome = None

    def add_children(self, children: dict) -> None:
        """
        Add a list of nodes to the children of this node.
        """
        for child in children:
            self.children[child.action] = child

    @property
    def value(self, explore: float = 0.5):
        """
        Calculate the UCT value of this node relative to its parent, the parameter
        "explore" specifies how much the value should favor nodes that have
        yet to be thoroughly explored versus nodes that seem to have a high win
        rate.
        """
        # if the node is not visited, set the value as infinity.
        if self.N == 0:
            return 0 if explore == 0 else GameMeta.INF
        else:
            avg = self.Q / self.N
            variance = avg * (1 - avg)
            return avg + explore * math.sqrt(math.log(self.parent.N) / self.N) * min(0.25, variance + math.sqrt(
                2 * math.log(self.parent.N) / self.N))

class GameState:
    """
    Stores information representing the current state of a game. 
    Also provides functions for playing game.
    """
    # dictionary associating numbers with players
    # PLAYERS = {"none": 0, "me": 1, "adv": 2}
    # I'm always player 1, adversay is always player 2

    def __init__(self, chess_board, my_pos, adv_pos, max_step):
        """
        Initialize the game board and give white first turn.
        Also create our union find structures for win checking.
        """
        self.chess_board = chess_board 
        self.board_size = chess_board.shape[0]
        self.to_play = GameMeta.PLAYERS['me']
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        is_endgame, my_score, adv_score = self.check_endgame()
        if is_endgame:
            self.winner = self.determine_winner(my_score, adv_score)
        else:
            self.winner = None
    
    def determine_winner(self, my_score, adv_score):
        # The game is over
        # I just played
        if self.to_play == GameMeta.PLAYERS["adv"]:
            if my_score > adv_score:
                return GameMeta.PLAYERS["me"]
            elif adv_score > my_score:
                return GameMeta.PLAYERS["adv"]
            else:
                return GameMeta.PLAYERS["none"]
        # Adv just played
        elif self.to_play == GameMeta.PLAYERS["me"]:
            if adv_score > my_score:
                return GameMeta.PLAYERS["adv"]
            elif my_score > adv_score:
                return GameMeta.PLAYERS["me"]
            else:
                return GameMeta.PLAYERS["none"]
        
        print("ERROR - SHOULDN'T REACH HERE AS A WINNER SHOULD BE ASSIGNED")
        return None
        

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
                    GameMeta.MOVES[1:3]
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
        p1_r = find(tuple(self.my_pos))
        p2_r = find(tuple(self.adv_pos))
        p1_score = list(father.values()).count(p1_r)
        p2_score = list(father.values()).count(p2_r)
        if p1_r == p2_r:
            return False, p1_score, p2_score

        return True, p1_score, p2_score

    def play(self, action: tuple) -> None:
        """
        Make an action and switch player turns
        Args:
            action (tuple): (pos, dir) = ((r, c), dir)
        """
        next_pos, dir = action

        # Set the barrier to True
        r, c = next_pos
        self.set_barrier(r, c, dir)

        # Update player position and switch turns
        if self.to_play == GameMeta.PLAYERS['me']:
            self.my_pos = next_pos
            self.to_play = GameMeta.PLAYERS['adv']
        else:
            self.adv_pos = next_pos
            self.to_play = GameMeta.PLAYERS['me']

        is_endgame, my_score, adv_score = self.check_endgame()
        if is_endgame:
            self.winner = self.determine_winner(my_score, adv_score)
        else:
            self.winner = None
        
    def set_barrier(self, r, c, dir):
        # Set the barrier to True
        self.chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = GameMeta.MOVES[dir]
        self.chess_board[r + move[0], c + move[1], GameMeta.OPPOSITES[dir]] = True

    def turn(self) -> int:
        """
        Return the player with the next action move.
        """
        return self.to_play

    def set_turn(self, player: int) -> None:
        """
        Set the player to take the next action.
        Raises:
            ValueError if player turn is not 1 or 2
        """
        if player in GameMeta.PLAYERS.values() and player != GameMeta.PLAYERS['none']:
            self.to_play = player
        else:
            raise ValueError('Invalid turn: ' + str(player))

    def actions(self) -> list:
        """
        Get a list of all actions possible on the current board.
        """
        actions = []

        # We can add barriers around ourselves if one does not exist in that direction yet
        if self.to_play == GameMeta.PLAYERS["me"]:
            ori_pos = self.my_pos
            opp_ori_pos = self.adv_pos
        else:
            ori_pos = self.adv_pos
            opp_ori_pos = self.my_pos
        ori_r, ori_c = ori_pos
        for dir in range(4):
            if not self.chess_board[ori_r, ori_c, dir]:
                actions.append(tuple([(ori_r, ori_c), dir]))

        # BFS
        state_queue = [(ori_pos, 0)]
        visited = {ori_pos}
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for move_dir, move in enumerate(GameMeta.MOVES):
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if self.chess_board[r, c, move_dir] or tuple_equal(next_pos, opp_ori_pos) or next_pos in visited:
                    continue

                next_pos_r, next_pos_c = next_pos
                for barrier_dir in range(4):
                    if not self.chess_board[next_pos_r, next_pos_c, barrier_dir]:
                        actions.append(tuple([next_pos, barrier_dir]))
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return actions

def tuple_equal(t1: tuple, t2: tuple):
    return (t1[0] == t2[0] and t1[1] == t2[1])

class GameMeta:
    PLAYERS = {'none': 0, 'me': 1, 'adv': 2}
    INF = float('inf')
    MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
    OPPOSITES = {0: 2, 1: 3, 2: 0, 3: 1}