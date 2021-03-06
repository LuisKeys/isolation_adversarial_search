from isolation import Isolation, DebugState
from sample_players import DataPlayer
import math
import random

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation
    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.
    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    _total_iterations = 0
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least
        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.
        See RandomPlayer and GreedyPlayer in sample_players for more examples.
        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 2:
            if 57 in state.actions():
                self.queue.put(57)                
            else:
                self.queue.put(random.choice(state.actions()))
        else:
          last_action = None
          for depth in range(1, 4):
            actions = self.alpha_beta_search(state, depth)
            self.queue.put(actions)
            #dbstate = DebugState.from_state(state)

    def alpha_beta_search(self, state, depth):

        def min_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state, depth)
            min_val = float("inf")
            for action in state.actions():
                min_val = min(min_val, max_value(state.result(action), alpha, beta, depth-1))
            return min_val

        def max_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state, depth)
            max_val = float("-inf")
            for action in state.actions():
                max_val = max(max_val, min_value(state.result(action), alpha, beta, depth-1))
            return max_val

        def get_best_action(action, next_action):
            (best_move, alpha, best_score) = action
            value = min_value(state.result(next_action), alpha, float("-inf"), depth-1)
            alpha = max(alpha, value)
            
            if value >= best_score:
                result = (next_action, alpha, value)
            else: 
                result = (best_move, alpha, best_score)

            return result
        
        next_move = (None, float("-inf"), float("-inf"))

        count = 0
        actions_num = len(state.actions())
        for action in state.actions():
            if count < actions_num / 2:
                self._total_iterations += 1
                next_move = get_best_action(next_move, action)
                count += 1
        
        return next_move[0]

    def ind2xy(self, ind):
        return (ind % (11), ind // (11))

    def score(self, state, depth):
        cust_agent_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        cust_agent_liberties = state.liberties(cust_agent_loc)
        opp_liberties = state.liberties(opp_loc)
        cust_x, cust_y = self.ind2xy(cust_agent_loc)
        distance = 7 - math.sqrt((cust_x - 4)**2 + (cust_y - 5)**2)
        result = distance * (len(cust_agent_liberties) - 2 * len(opp_liberties))
        return result
