import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

# Important: you should register your agent with a name
@register_agent("offence_random_agent")
class OffenceRandomAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(OffenceRandomAgent, self).__init__()
        self.name = "OffenceRandomAgent"
        self.autoplay = True

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    # If yes, return cell and direction
    # If not, return -1
    def canAdvBeBlocked(self, chess_board, adv_pos):
        num_walls_around_adv = 0
        adv_r, adv_c = adv_pos
        missing_wall_dir = -1
        for dir in range(4):
            if chess_board[adv_r, adv_c, dir]:
                num_walls_around_adv += 1
            else: 
                missing_wall_dir = dir
        if(num_walls_around_adv == 3):
            adj_cell_r = adv_r + self.moves[missing_wall_dir][0]
            adj_cell_c = adv_c + self.moves[missing_wall_dir][1]
            adj_dir = (missing_wall_dir+2)%4
            return (adj_cell_r, adj_cell_c, adj_dir)
        else:
            return (-1, -1, -1)

    # If opponent has 3 walls around it and I can move there, go and block them
    # Else make a random move
    def step(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        blocking_cell_r, blocking_cell_c, blocking_dir = self.canAdvBeBlocked(chess_board, adv_pos)
        if(blocking_cell_r != -1): 
            print("Adv can be blocked, trying to go there")
            checkIfValidStep = self.check_valid_step(chess_board, np.array(my_pos), tuple([blocking_cell_r, blocking_cell_c]), blocking_dir, adv_pos, max_step)
            if(checkIfValidStep):
                print("Valid step, going to block")
                return (blocking_cell_r, blocking_cell_c), blocking_dir
            else:
                print("could not reach :(")
        
        print("Adv cannot be blocked, take random step")
        steps = np.random.randint(0, max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    # From world.py
    def check_valid_step(self, chess_board, my_start_pos, goal_end_pos, goal_barrier_dir, adv_pos, max_step):
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
        goal_r, goal_c = goal_end_pos
        if chess_board[goal_r, goal_c, goal_barrier_dir]:
            return False
        if self.pos_equal(my_start_pos, goal_end_pos): # equivalent to np.array_equal(start_pos, end_pos)
            return True

        # Get position of the adversary
        # adv_pos = self.p0_pos if self.turn else self.p1_pos

        # BFS
        state_queue = [(my_start_pos, 0)]
        visited = {tuple(my_start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if self.pos_equal(next_pos, adv_pos) or tuple(next_pos) in visited: # np.array_equal(next_pos, adv_pos)
                    continue
                if self.pos_equal(next_pos, goal_end_pos): # np.array_equal(next_pos, end_pos)
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def pos_equal(self, pos1, pos2):
        return (pos1[0] == pos2[0] and pos1[1] == pos2[1])
