from isolation import Isolation, DebugState
from sample_players import DataPlayer
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
            #print(dbstate)

    def alpha_beta_search(self, state, depth):

        def min_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            min_val = float("inf")
            for action in state.actions():
                min_val = min(min_val, max_value(state.result(action), alpha, beta, depth-1))
            return min_val

        def max_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
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

        for action in state.actions():
            next_move = get_best_action(next_move, action)
        
        return next_move[0]

    def score(self, state):
        cust_agent_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        cust_agent_liberties = state.liberties(cust_agent_loc)
        opp_liberties = state.liberties(opp_loc)
        partition_score = self.check_partition(cust_agent_loc, opp_loc, state)        
        result = len(cust_agent_liberties) - len(opp_liberties)
        #result = partition_score * len(cust_agent_liberties) - 2 * len(opp_liberties)
        #result = 0
        return result

    def check_partition(self, cust_agent_loc, opp_loc, opp_liberties):
        cust_x, cust_y = DebugState.ind2xy(cust_agent_loc)
        opp_x, opp_y = DebugState.ind2xy(opp_loc)

        result = 1

        if opp_x > cust_x and opp_x - cust_x < 3 and cust_x > 6:
            result += self.get_blocked(cust_x, cust_y, opp_liberties, True, False, False, False)

        if opp_x < cust_x and cust_x - opp_x < 3 and cust_x < 2:
            result += self.get_blocked(cust_x, cust_y, opp_liberties, False, True, False, False)

        if opp_y > cust_y and opp_y - cust_y < 3 and cust_y > 8:
            result += self.get_blocked(cust_x, cust_y, opp_liberties, False, False, True, False)

        if opp_y < cust_y and cust_y - opp_y < 3 and cust_y < 2:
            result += self.get_blocked(cust_x, cust_y, opp_liberties, False, False, False, True)

        
        return result
    
    def get_blocked(self, cust_x, cust_y, opp_liberties, left, right, top, bottom):

        blocked = True
        for liberty in opp_liberties.locs:
            opp_x, opp_y = DebugState.ind2xy(liberty)

            if left and opp_x <= cust_x:
                blocked = False
                break

            if right and opp_x >= cust_x:
                blocked = False
                break

            if top and opp_y >= cust_y:
                blocked = False
                break

            if bottom and opp_y <= cust_x:
                blocked = False
                break

        if blocked:
            return 1

        return 0
