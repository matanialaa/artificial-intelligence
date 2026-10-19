from numpy import random

from mdp import Action, MDP
import copy
from copy import deepcopy
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np
def get_state_policy_and_utility(mdp: MDP, state, utility):
    """
    For a given state and utility matrix, returns the best action and its expected utility.
    """
    def q_value(action):
        return get_sum_of_probabilities(mdp, state, utility, action)

    actions = mdp.actions
    best_action = max(actions, key=q_value)
    best_action_value = q_value(best_action)
    return best_action, best_action_value
def get_state_policy(mdp: MDP, state, utility):
    actions = mdp.actions
    best_action = "up"
    best_action_value = float('-inf')
    for action in actions:
        cur_action_value = get_sum_of_probabilities(mdp, state, utility, action)
        if best_action_value < cur_action_value:
            best_action_value = cur_action_value
            best_action = action
    return best_action
def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #

    U_final = None
    # TODO:
    new_utility = U_init
    # states:
    row_num = mdp.num_row
    col_num = mdp.num_col
    states = [(i, j) for i in range(row_num) for j in range(col_num)]
    # gamma:
    gamma = mdp.gamma
    # states with no updates to utility (walls and goal states):
    goal_states = mdp.terminal_states
    walls = []
    for state in states:
        if mdp.board[state[0]][state[1]] == 'WALL':
            walls.append(state)
    # actions:
    actions = mdp.actions
    # when to stop the function:
    termination_cond = (epsilon * (1 - gamma)) / gamma

    while True:
        d = 0  # delta
        prev_utility = copy.deepcopy(new_utility)
        for state in states:
            if state in walls:
                continue
            if state in goal_states:
                new_utility[state[0]][state[1]] = float(mdp.board[state[0]][state[1]])
                continue
            # Sum over s' for P(s'|a,s)*U(s'):
            all_prob_utils = [get_sum_of_probabilities(mdp, state, prev_utility, a) for a in actions]
            # max Sum ( P(s'|a,s)*U(s') ):
            max_prob_util = max(all_prob_utils)
            state_reward = float(mdp.board[state[0]][state[1]])  # rewards are saved as str in the board
            # U(s) = R(s) + gamma*max(Sum ( P(s'|a,s)*U(s') ))
            state_util = state_reward + gamma * max_prob_util
            new_utility[state[0]][state[1]] = state_util
            # update delta
            diff = abs(prev_utility[state[0]][state[1]] - new_utility[state[0]][state[1]])
            d = max(d, diff)
        if d < termination_cond:
            break
    # ========================
    return new_utility


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 
    row_num = mdp.num_row
    col_num = mdp.num_col
    states = [(i, j) for i in range(row_num) for j in range(col_num)]
    policy = [[] for row in range(row_num)]
    for row in range(row_num):
        policy[row] = ['up' for i in range(col_num)]
    # states with no updates to utility (walls and goal states):
    goal_states = mdp.terminal_states
    walls = []
    for state in states:
        if mdp.board[state[0]][state[1]] == 'WALL':
            walls.append(state)
    for state in goal_states:
        policy[state[0]][state[1]] = mdp.board[state[0]][state[1]]
    for state in walls:
        policy[state[0]][state[1]] = mdp.board[state[0]][state[1]]
    for state in states:
        if state in goal_states:
            continue
        if state in walls:
            continue
        action = get_state_policy(mdp, state, U)
        policy[state[0]][state[1]] = action

    # ========================
    return policy

def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    U = None
    # TODO:
    # ====== YOUR CODE: ======
    # states:
    row_num = mdp.num_row
    col_num = mdp.num_col
    states = [(i, j) for i in range(row_num) for j in range(col_num)]
    utility = [[] for row in range(row_num)]
    for row in range(row_num):
        utility[row] = [0 for i in range(col_num)]
    # states with no updates to utility (walls and goal states):
    goal_states = mdp.terminal_states
    walls = []
    for state in states:
        if mdp.board[state[0]][state[1]] == 'WALL':
            walls.append(state)
    for state in goal_states:
        utility[state[0]][state[1]] = float(mdp.board[state[0]][state[1]])

    d, first = 0, True
    # sort of like ' do .. while()' instead of ' while() do .. ' #####THANK YOU MAROON FOR THIS IDEA
    while first or d > 0:
        d, first = 0, False
        for state in states:
            row = state[0]
            col = state[1]
            if state in walls:
                continue
            if state in goal_states:
                continue
            prev_ut = utility[row][col]
            new_sum_prob_ut = get_state_cur_utility(mdp, utility, state, policy[row][col])
            new_ut = float(mdp.board[row][col]) + mdp.gamma * new_sum_prob_ut
            utility[row][col] = new_ut
            d = max([d, abs(new_ut - prev_ut)])
    # ========================
    return utility
def get_next_state(mdp: MDP, state: tuple, action: str):
    row, col = state
    if action == "up":
        if row >= 1:
            if not mdp.board[row - 1][col] == "WALL":
                return (row - 1, col)
        else:
            return (row, col)
    if action == "down":
        if row < mdp.num_row - 1:
            if not mdp.board[row + 1][col] == "WALL":
                return (row + 1, col)
        else:
            return (row, col)
    if action == "right":
        if col < mdp.num_col - 1:
            if not mdp.board[row][col + 1] == "WALL":
                return (row, col + 1)
        else:
            return (row, col)
    if action == "left":
        if col >= 1:
            if not mdp.board[row][col - 1] == "WALL":
                return (row, col - 1)
        else:
            return (row, col)
    return (row, col)
def get_sum_of_probabilities(mdp: MDP, state: tuple, prev_utilities, action):
    row, col = state
    res = 0
    actions = ["up", "down", "right", "left"]
    next_states = [get_next_state(mdp, state, a) for a in actions]
    # note to self: next states holds the next states in the order they appear in actions
    probabilities = [(i, mdp.transition_function[action][i]) for i in range(4)]
    # ls = enumerate(mdp.transition_function[action])
    for i, prob_i in probabilities:
        next_state = next_states[i]
        if next_state == None:
            continue
        next_row = next_state[0]
        next_col = next_state[1]
        res += prob_i * prev_utilities[next_row][next_col]
    return res
def get_state_cur_utility(mdp: MDP, utility, state, action):
    if isinstance(action, str):
        action = Action[action.upper()]
    st_ut = 0.0
    actions = mdp.actions
    #action is the step we want to take, actual_action is the step actually taken which is not deterministic
    for i, actual_action in enumerate(actions):
        next_state = mdp.step(state, actual_action)
        prob = mdp.transition_function[action][i]
        next_state_ut = utility[next_state[0]][next_state[1]]
        st_ut += prob*next_state_ut
    return st_ut

def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    row_num = mdp.num_row
    col_num = mdp.num_col
    states = [(i, j) for i in range(row_num) for j in range(col_num)]
    goal_states = mdp.terminal_states
    walls = []
    for state in states:
        if mdp.board[state[0]][state[1]] == 'WALL':
            walls.append(state)

    # policy iteration alg:
    changed = True
    cur_policy = copy.deepcopy(policy_init)

    while changed:
        utility = policy_evaluation(mdp, cur_policy)
        changed = False
        for state in states:
            if state in walls:
                continue
            if state in goal_states:
                continue
            row = state[0]
            col = state[1]
            best_action, best_action_value = get_state_policy_and_utility(mdp, state, utility)

            st_ut = get_state_cur_utility(mdp, utility, state, cur_policy[row][col])

            if best_action_value > st_ut:
                cur_policy[row][col] = best_action
                changed = True
    # ========================
    return cur_policy


def mc_algorithm(
        sim,
        num_episodes,
        gamma,
        num_rows=3,
        num_cols=4,
        actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT],
        policy=None,
):
    # Given a simulator, the number of episodes to run, the number of rows and columns in the MDP, the possible actions,
    # and an optional policy, run the Monte Carlo algorithm to estimate the utility of each state.
    # Return the utility of each state.

    V = None
    V = np.zeros((num_rows, num_cols), dtype=float)

    returns_sum = {}
    returns_count = {}

    # Replay episodes from file
    for episode_gen in sim.replay(num_episodes):
        episode = list(episode_gen)  # convert generator to list

        G = 0.0
        visited = set()

        # go backwards
        for state, reward, action, actual_action in reversed(episode):
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                returns_sum[state] = returns_sum.get(state, 0.0) + G
                returns_count[state] = returns_count.get(state, 0) + 1
    for state in returns_sum:
        r, c = state
        V[r, c] = returns_sum[state] / returns_count[state]

    return V
