from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import math
import time



def termination_condition(env: WarehouseEnv, depth, kill_time):
    if time.time() >= kill_time:
        return True
    if depth <= 0:
        return True
    if env.done():
        return True
    return False

def get_op_succ_tup(env: WarehouseEnv, whose_turn):
    operators = env.get_legal_operators(whose_turn)
    list_of_op_succ_tups = []
    for op in operators:
        cloned_env = env.clone()
        cloned_env.apply_operator(whose_turn, op)
        tup = (op, cloned_env)
        list_of_op_succ_tups.append(tup)
    for tup in list_of_op_succ_tups:
        if tup[1] is None:
            raise NotImplementedError()
    return list_of_op_succ_tups



# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    opponent_id = 0
    robot = env.get_robot(robot_id)
    if robot_id == 0:
        opponent_id = 1
    opponent = env.get_robot(opponent_id)
    if env.done():
        return robot.credit - opponent.credit
    charge_distinaton = [manhattan_distance(robot.position, charge.position) for charge in env.charge_stations]
    best_c = min(charge_distinaton, default=0)
    distination = [manhattan_distance(robot.position, package.position) for package in env.packages if package.on_board]
    best_pd = min(distination, default=0)
    if robot.package is None:

        if robot.credit > 0 and robot.battery <= best_c and robot.credit < 2 * opponent.credit:
            return 25 - best_c + robot.battery
        return robot.battery + robot.credit * 25 - best_pd
    else:
        if robot.battery <= best_c and robot.credit < 2 * opponent.credit:
            return 25 - best_c + robot.battery
        return 25 - manhattan_distance(robot.position, robot.package.destination) + robot.battery + robot.credit * 25

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

def termination_cond(env: WarehouseEnv, depth, time):
    if  time <= 0.1:
        return True
    if depth == 0:
        return True
    if env.done():
        return True
    return False
class AgentMinimax(Agent):
    # TODO: section b : 4
    def time_left1(self, begin: float, time_limit: float) -> float:
        return time_limit - (time.time() - begin)
    def minimax_iteration(self, env: WarehouseEnv, begin, agent_id, depth, time_left, time_limit, turn):
        if termination_cond(env,depth,time_left):
            return smart_heuristic(env, agent_id)
        operators, children = self.successors(env, agent_id)
        minimum = turn % 2
        curr_max = float('-inf')
        curr_min = float('inf')
        for child in children:
            time_left = self.time_left1(begin,time_limit)
            heuristic = self.minimax_iteration(child, begin, agent_id, depth - 1, time_left, time_limit, turn + 1)
            curr_max = max(curr_max, heuristic)
            curr_min = min(curr_min, heuristic)
        if minimum == 1:
            return curr_min
        return curr_max

    def minimax_anytime(self, env, agent_id, D, time_limit, begin, time_left):
        operators, children = self.successors(env, agent_id)
        children_minmax = [self.minimax_iteration(child, begin, agent_id, D, time_left, time_limit, 0) for child in children
        ]

        if not children_minmax:
            return None
        max_op = max(children_minmax)
        opp = operators[children_minmax.index(max_op)]
        return opp

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        begin = time.time()
        depth = 0
        time_left = time_limit
        operators = env.get_legal_operators(agent_id)
        op = random.choice(operators)

        while time_left >= 0.1 and depth <= 5:
            iteration = self.minimax_anytime(env, agent_id, depth, time_limit, begin, time_left)
            if iteration:
                op = iteration
            time_left = self.time_left1(begin,time_limit)
            depth += 1
        return op


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def next_turn(self, cur_turn):
        return int(not cur_turn)

    def alpha_beta_iteration(self, env: WarehouseEnv, whose_turn, depth, a, b, kill_time):
        if termination_condition(env, depth, kill_time):
            return smart_heuristic(env, whose_turn)

        op_suc = get_op_succ_tup(env, whose_turn)

        temp = 1
        if whose_turn != self.agent:
            temp = -1
        curMax = float('-inf')
        curMin = float('inf')
        for op, cloned_env in op_suc:
            if time.time() > kill_time:
                return smart_heuristic(cloned_env, whose_turn)
            next_turn = self.next_turn(whose_turn)
            v = self.alpha_beta_iteration(cloned_env, next_turn, depth - 1, a, b, kill_time)
            if temp == 1:
                if v > curMax:
                    curMax = v
                a = max(curMax, a)
                if curMax >= b:
                    return float('inf')
            else:
                if v < curMin:
                    curMin = v
                b = min(curMin, b)
                if curMin <= a:
                    return float('-inf')
        if temp == 1:
            return curMax
        else:
            return curMin

    def alpha_beta_anytime(self, env: WarehouseEnv, agent, kill_time):
        cur_max = float('-inf')
        depth = 1
        cris = 10
        messi = 9
        envi = env.clone()
        ops = envi.get_legal_operators(agent)
        operator = None
        operator = "move north"

        while (cris > messi and time.time() <= kill_time):
            local_max = float('-inf')
            op_suc = get_op_succ_tup(envi, agent)
            local_op = op_suc[0][0]
            cris = cris + 1
            for op, cloned_env in op_suc:
                if time.time() > kill_time and cris > messi:
                    return operator
                v = self.alpha_beta_iteration(cloned_env, agent, depth, float('-inf'), float('inf'), kill_time)
                if v >= local_max and cris > messi:
                    local_max = v
                    cris = cris + 1
                    local_op = op
            if local_max >= cur_max:
                cur_max = local_max
                cris = cris + 1
                operator = local_op
            if cris > cris + 1:
                messi = messi + 1
            depth = depth + 1

        return operator

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent = agent_id
        kill_time = time.time() + (0.98) * time_limit

        return self.alpha_beta_anytime(env.clone(), agent_id, kill_time)




def helper_expectimax(env, player_id, current_turn, depth_limit, move_to_return, timeout):
    if time.time() >= timeout:
        raise ValueError("time is Up")
    if termination_condition(env, depth_limit, timeout):
        return(move_to_return, smart_heuristic(env, player_id))

    steps = env.get_legal_operators(current_turn)
    if not steps:
        return (None, smart_heuristic(env, player_id))
    next_states = []
    for act in steps:
        sim_env = env.clone()
        sim_env.apply_operator(current_turn, act)
        next_states.append(sim_env)

    selected_move = None

    if current_turn == player_id:
        max_score = -math.inf
        for sim, act in zip(next_states, steps):
            _, result = helper_expectimax(sim, player_id, (current_turn + 1) % 2, depth_limit - 1, act, timeout)
            if result > max_score:
                max_score = result
                selected_move = act
        return (selected_move, max_score)
    else:
        regular_prob = 1 / (len(steps) + 4)
        favorite = 3 * regular_prob
        expected_value = 0
        for sim, act in zip(next_states, steps):
            _, val = helper_expectimax(sim, player_id, (current_turn + 1) % 2, depth_limit - 1, None, timeout)
            if act in ['move west', 'pick up']:
                expected_value += favorite * val
            else:
                expected_value += regular_prob * val
        return (None, expected_value)





class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        end_time = time.time() + time_limit - 0.1
        final_choice = None
        depth = 0

        try:
            while depth <= 5:
                if time.time() > end_time:
                    raise ValueError("time is Up")
                suggestion, _ = helper_expectimax(env, agent_id, agent_id, depth, None, end_time)
                available_moves = env.get_legal_operators(agent_id)
                if suggestion in available_moves:
                    final_choice = suggestion
                depth += 1
        except ValueError:
            pass

        fallback_ops = env.get_legal_operators(agent_id)
        return final_choice if final_choice in fallback_ops else random.choice(fallback_ops)














# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)