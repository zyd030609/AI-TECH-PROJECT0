"""蒙特卡洛树搜索"""

import torch
import numpy as np
import copy
from config import CONFIG
from torch.amp import autocast
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import time

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
        #
        actions, values = zip(*[(act, float(child.get_value(c_puct))) for act, child in self._children.items()])
        actions = torch.tensor(actions, device='cuda')
        values = torch.tensor(values, device='cuda')

        # 使用 PyTorch 的 torch.argmax 来获取最大值索引
        max_index = torch.argmax(values)
        # 返回动作和下一个节点
        return actions[max_index].cpu().item(), self._children[actions[max_index].cpu().item()]

    #
    def get_value(self, c_puct):
        """计算并返回此节点的值"""
        if self._parent is None:
            raise ValueError("TreeNode.get_value called on root node!")

        if self._parent._n_visits == 0:
            #print("Warning: Parent node has zero visits!")
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
    def __init__(self, policy_value_fn, c_puct=4, n_playout=2000,if_visualize=False):
        """
        :param policy_value_fn: 一个函数，接收棋盘状态并返回动作概率和状态价值
        :param c_puct: 控制探索和利用平衡的超参数
        :param n_playout: 每次模拟的次数
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn  # 接收经过包装的 policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._if_visualize = if_visualize

    def init_visualize(self):
        """初始化streamlit和networkx"""
        st.title("MCTS树结构可视化")
        tree_container = st.empty()
        return tree_container

    def visualize(self, tree_container,if_over=False):
        # 收集每一层的节点
        rows = {}  # key:int，value:[TreeNode]，用于记录每层要显示的节点
        rows[0] = [self._root]
        chosen_nodes=[]
        i = 0
        while 1:
            if i==0:
                chosen_node=rows[i][0]
                chosen_nodes.append(chosen_node)
            else:
                if if_over:
                    chosen_node=max(rows[i], key=lambda x: x._n_visits)
                else:
                    chosen_node=max(rows[i], key=lambda x: x.get_value(self._c_puct))
                chosen_nodes.append(chosen_node)
            if chosen_node._children:
                rows[i+1]=[]
                rows[i+1].extend(list(chosen_node._children.values()))
            else:
                break
            i += 1

        # 创建有向图
        G = nx.DiGraph()
        
        # 计算节点位置
        pos = {}
        for level in rows:
            if_more=False
            if len(rows[level])>7 and level>0:
                if_more=True
                more=len(rows[level])-7
            # 根据 _Q + _u 的值选择前7个节点
            if if_more:
                # 找出前7个最大值的索引，但不改变顺序
                if if_over:
                    values = [(i, x._n_visits) for i, x in enumerate(rows[level])]
                else:
                    values = [(i, x.get_value(self._c_puct)) for i, x in enumerate(rows[level])]
                top_indices = sorted(range(len(values)), key=lambda i: values[i][1], reverse=True)[:7]
                top_indices = sorted(top_indices)  # 按原始顺序排序索引
                row = [rows[level][i] for i in top_indices]
                row.append(TreeNode(row[0]._parent, 1.0))
            else:
                row = rows[level]
            
            # 计算当前层的水平间距
            level_width = max(1.0, len(row) - 1)
            for j, node in enumerate(row):
                # x 坐标在 [-0.5, 0.5] 范围内均匀分布
                x = -0.5 + j / level_width if level_width > 0 else 0
                y = -level  # y 坐标表示层级，向下递减
                pos[node] = (x, y)
                
                # 添加节点和边
                if isinstance(node._Q, (list, np.ndarray)):  # 如果是数组或列表
                    q_value = node._Q[0] if isinstance(node._Q, list) else node._Q.item()  # 取第一个值或转换为标量
                else:
                    q_value = node._Q  # 如果是标量，直接使用
                if isinstance(node._u, (list, np.ndarray)):  # 如果是数组或列表
                    u_value = node._u[0] if isinstance(node._u, list) else node._u.item()  # 取第一个值或转换为标量
                else:
                    u_value = node._u  # 如果是标量，直接使用
                if isinstance(node._P, (list, np.ndarray)):
                    p_value = node._P[0] if isinstance(node._P, list) else node._P.item()
                else:
                    p_value = node._P
                if j==7:
                    visits_str=f'{more}more'
                else:
                    visits_str = f'v={node._n_visits}\nQ={q_value:.2f}\nu={u_value:.2f}\nP={p_value:.2f}'
                
                if node not in chosen_nodes:
                    G.add_node(node, label=visits_str, color='lightblue')
                else:
                    G.add_node(node, label=visits_str, color='#FFE4E1', edgecolors='black', linewidth=1)
                if node._parent and node._parent in pos:  # 确保父节点存在
                    action = None
                    for act, child in node._parent._children.items():
                        if j==7:
                            action='...'
                            break
                        if child == node:
                            action = act
                            break
                    if node not in chosen_nodes:
                        G.add_edge(node._parent, node, label=str(action),color='lightgray')
                    else:
                        G.add_edge(node._parent, node, label=str(action),color='black')

        # 绘制图形
        plt.figure(figsize=(10, 2+len(rows)))
        nx.draw(G, pos=pos, 
                with_labels=True,
                node_color=[G.nodes()[node]['color'] for node in G.nodes()],
                edgecolors=[G.nodes()[node].get('edgecolors', 'none') for node in G.nodes()],
                linewidths=[G.nodes()[node].get('linewidth', 0) for node in G.nodes()],
                node_size=1500,
                arrows=True,
                edge_color=[G[u][v]['color'] for u,v in G.edges()],
                labels=nx.get_node_attributes(G, 'label'))
        
        # 添加边标签
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        tree_container.pyplot(plt.gcf(), clear_figure=True)
        plt.close('all')

    #单次的playout，指走一遍策略树，然后扩展一次叶节点，然后回溯更新
    def  _playout(self, state,tree_container=None):
        node = self._root
        i=0
        while True:
            if node.is_leaf():
                #将当前的状态传入神经网络，获取动作概率和状态价值
                action_probs, leaf_value = self._policy(state)
                if not action_probs:
                    print("No actions to expand for this node!")
                    break
                #扩展新的叶节点
                node.expand(action_probs)
                i+=1
                if self._if_visualize:
                    self.visualize(tree_container,if_over=False)
                #回溯更新：
                node.update_recursive(-leaf_value)
                return
            #如果不是叶节点，就选择一个动作，然后转移到下一个节点
            try:
                action, node = node.select(self._c_puct)
                i+=1
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

    #执行n_playout次playout，然后获取动作和动作概率
    def get_move_probs(self, state, temp=1e-3):
        """获取移动概率"""
        if self._if_visualize:
            tree_container = self.init_visualize()
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            if self._if_visualize:
                self._playout(state_copy,tree_container) 
                if n==self._n_playout-1:
                    self.visualize(tree_container,if_over=True)
            else:
                self._playout(state_copy) 
        
        # 计算树的最大深度
        def get_max_depth(node, current_depth=0):
            if not node._children:
                return current_depth
            return max(get_max_depth(child, current_depth + 1) for child in node._children.values())

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

    def __init__(self, policy_value_function, c_puct=4, n_playout=1000, is_selfplay=0,if_visualize=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout,if_visualize)
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
        acts, probs = self.mcts.get_move_probs(board, temp,)
        move_probs[list(acts)] = probs
        #选择并执行动作
        if self._is_selfplay:
            # 添加 Dirichlet Noise 进行探索（自我对弈需要）
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs))))
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
