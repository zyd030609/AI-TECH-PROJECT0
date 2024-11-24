"""蒙特卡洛树搜索"""

import torch
import numpy as np
import copy
from config import CONFIG
from torch.amp import autocast

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 定义叶子节点
class TreeNode(object):
    """
    mcts树中的节点，树的子节点字典中，键为动作，值为TreeNode。记录当前节点选择的动作，以及选择该动作后会跳转到的下一个子节点。
    每个节点跟踪其自身的Q，先验概率P及其访问次数调整的u
    """

    def __init__(self, parent, prior_p):
        """
        :param parent: 当前节点的父节点
        :param prior_p:  当前节点被选择的先验概率
        """
        self._parent = parent
        self._children = {} # 从动作到TreeNode的映射
        self._n_visits = 0  # 当前当前节点的访问次数
        self._Q = 0         # 当前节点对应动作的平均动作价值
        self._u = 0         # 当前节点的置信上限         # PUCT算法
        self._P = prior_p

    def expand(self, action_priors):
        """扩展树，生成新子节点"""
        if not action_priors:
            print("Warning: No action priors provided for expansion!")
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在 GPU 上选择值最高的动作"""
        if not self._children:
            raise ValueError("TreeNode.select called on a node with no children!")

        actions, values = zip(*[(act, float(child.get_value(c_puct))) for act, child in self._children.items()])
        actions = torch.tensor(actions, device='cuda')
        values = torch.tensor(values, device='cuda')

        # 使用 PyTorch 的 torch.argmax 来获取最大值索引
        max_index = torch.argmax(values)
        return actions[max_index].cpu().item(), self._children[actions[max_index].cpu().item()]

    def get_value(self, c_puct):
        """计算并返回此节点的值"""
        if self._parent is None:
            raise ValueError("TreeNode.get_value called on root node!")

        if self._parent._n_visits == 0:
            print("Warning: Parent node has zero visits!")
            return self._Q  # 或者返回一个默认值

        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """
        从叶节点评估中更新节点值
        leaf_value: 这个子节点的评估值来自当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值，取决于所有访问次数的平均树，使用增量式更新方式
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    # 使用递归的方法对所有节点（当前节点对应的支线）进行一次更新
    def update_recursive(self, leaf_value):
        """就像调用update()一样，但是对所有直系节点进行更新"""
        # 如果它不是根节点，则应首先更新此节点的父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """检查是否是叶节点，即没有被扩展的节点"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


# 蒙特卡洛搜索树
class MCTS:
    def __init__(self, policy_value_fn, c_puct=4, n_playout=800):
        """
        :param policy_value_fn: 一个函数，接收棋盘状态并返回动作概率和状态价值
        :param c_puct: 控制探索和利用平衡的超参数
        :param n_playout: 每次模拟的次数
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn  # 接收经过包装的 policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        while True:
            if node.is_leaf():
                action_probs, leaf_value = self._policy(state)
                if not action_probs:
                    print("No actions to expand for this node!")
                    break
                node.expand(action_probs)
                node.update_recursive(-leaf_value)
                return
            try:
                action, node = node.select(self._c_puct)
            except ValueError as e:
                print(f"Node select failed: {e}")
                break
            state.do_move(action)

    def policy_value_fn_batch(self, board_states):
        """
        批量推理多个棋盘状态。如果传入单个棋盘状态，也可以正常处理。
        """
        if not isinstance(board_states, list):  # 如果输入是单个状态
            board_states = [board_states]

        # 将所有棋盘状态转为 GPU 张量
        states = torch.stack([torch.tensor(board.current_state(), device=self.device) for board in board_states])

        with autocast('cuda'):  # 使用混合精度推理
            log_act_probs, values = self.policy_value_net(states)

        results = []
        for i, board in enumerate(board_states):
            legal_positions = torch.tensor(board.availables, device=self.device)
            act_probs = torch.exp(log_act_probs[i][legal_positions])
            results.append((list(zip(legal_positions.cpu().numpy(), act_probs.cpu().numpy())), values[i].cpu().numpy()))

        return results

    def policy_value_fn_batch(self, board_states):
        if not isinstance(board_states, list):  # 如果输入是单个状态
            board_states = [board_states]

        # 转为张量
        states = torch.stack([torch.tensor(board.current_state(), device='cuda') for board in board_states])
        with autocast('cuda'):
            log_act_probs, values = self.policy_value_net(states)

        # 直接在 GPU 上处理合法动作
        results = []
        for i, board in enumerate(board_states):
            legal_positions = torch.tensor(board.availables, device='cuda')
            act_probs = torch.exp(log_act_probs[i, legal_positions])
            results.append((legal_positions.cpu().numpy(), act_probs.cpu().numpy(), values[i].cpu().numpy()))
        return results

    def policy_value_fn(self, board):
        """单状态推理，利用批量推理实现"""
        return self.policy_value_fn_batch([board])[0]

    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 计算根节点的动作访问计数
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]

        if not act_visits:
            print("Warning: No valid moves available at root!")
            return [], []  # 直接返回空列表，避免解包错误

        # 如果有访问次数，则继续计算
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        在当前的树上向前一步，保持我们已经直到的关于子树的一切
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'


# 基于MCTS的AI玩家
class MCTSPlayer(object):

    def __init__(self, policy_value_function, c_puct=4, n_playout=1000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_ind(self, p):
        self.player = p

    # 重置搜索树
    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __str__(self):
        return 'MCTS {}'.format(self.player)

    # 得到行动
    def get_action(self, board, temp=1e-3, return_prob=0):
        # 像 AlphaGo Zero 论文一样使用 MCTS 算法返回的 pi 向量
        move_probs = np.zeros(2086)

        # 获取动作和概率
        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            # 添加 Dirichlet Noise 进行探索（自我对弈需要）
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )
            # 更新根节点并重用搜索树
            self.mcts.update_with_move(move)
        else:
            # 使用默认的 temp=1e-3，它几乎相当于选择具有最高概率的移动
            move = np.random.choice(acts, p=probs)
            # 重置根节点
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        else:
            return move
