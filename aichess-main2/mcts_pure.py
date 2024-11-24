# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)

@author: Junxiao Song
"""
import torch
import numpy as np
import copy
from operator import itemgetter
from torch.amp import autocast


def rollout_policy_fn(board):
    """启发式随机策略"""
    availables = list(board.availables)
    # 默认生成随机概率
    action_probs = torch.rand(len(availables), device='cuda')

    # 可以加入启发式权重（例如优先中心或特定棋盘规则）
    # 示例：假设中心位置动作优先
    center_bonus = torch.tensor([1.2 if is_center(action) else 1.0 for action in availables], device='cuda')
    action_probs *= center_bonus
    action_probs /= action_probs.sum()  # 归一化

    return zip(availables, action_probs.cpu().numpy())


def policy_value_fn(self, board):
    self.policy_value_net.eval()
    legal_positions = torch.tensor(board.availables, device=self.device)
    current_state = torch.as_tensor(board.current_state(), device=self.device)  # 直接转为 GPU 张量
    current_state = current_state.unsqueeze(0)  # 添加 batch 维度

    with autocast('cuda'):
        log_act_probs, value = self.policy_value_net(current_state)

    log_act_probs = log_act_probs.squeeze(0)  # 移除 batch 维度
    act_probs = torch.exp(log_act_probs)

    # 仅取合法动作的概率
    act_probs = act_probs[legal_positions]

    if len(legal_positions) == 0:  # 检查是否有合法动作
        print("No legal moves available!")

    return list(zip(legal_positions.cpu().numpy(), act_probs.cpu().numpy())), value.detach().cpu().numpy()




class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """扩展树，生成新子节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在 GPU 上选择值最高的动作"""
        if not self._children:
            raise ValueError("TreeNode.select called on a node with no children!")

        # 使用 GPU 计算最大值索引
        actions, values = zip(*[(act, child.get_value(c_puct)) for act, child in self._children.items()])
        actions = torch.tensor(actions, device='cuda')
        values = torch.tensor(values, device='cuda')
        max_index = torch.argmax(values)
        return actions[max_index].cpu().item(), self._children[actions[max_index].cpu().item()]

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值"""
        if self._parent is None:  # 根节点没有父节点，无法计算值
            raise ValueError("TreeNode.get_value called on root node!")

        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    # 在 _playout 方法中将节点选择、状态更新操作移至 GPU
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

    def _evaluate_rollout(self, state, limit=1000):
        player = state.get_current_player_id()
        for _ in range(limit):
            if state.is_terminal():  # 使用终局检查函数（需在 Board 类中实现 is_terminal）
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("WARNING: rollout reached move limit")

        end, winner = state.game_end()
        if winner == -1:
            return 0  # 平局
        return 1 if winner == player else -1

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            state_copy = copy.copy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTS_Pure(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        """返回最佳动作"""
        if board.is_terminal():  # 提前终止
            print("Game is already over.")
            return None

        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")
            return None

    def __str__(self):
        return "MCTS {}".format(self.player)
