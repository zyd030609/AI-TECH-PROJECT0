import matplotlib.pyplot as plt
from mcts import TreeNode, MCTSPlayer,MCTS
from pytorch_net import PolicyValueNet
from game import move_action2move_id, Game, Board
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import streamlit as st

# plt.ion() #开启交互模式，画面只在同一个窗口内变化

board_list_init = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]


# board=Board()
# board.init_board(1)
# policy_value_net = PolicyValueNet(model_file='current_policy.pkl')
# mcts=MCTS(policy_value_net.policy_value_fn)
# mcts.get_move_probs(board)
# root=mcts._root
# print(len(root._children))
# i=0
# while root._children:
#     max_child = max(root._children.items(), key=lambda x: x[1]._n_visits)[0]
#     root = root._children[max_child]
#     print(i)
#     print(root._n_visits)
#     print(root._Q)
#     print(root._u)
#     print(root._P)
#     print(root._children)
#     i+=1
# root=mcts._root


def hierarchical_pos(graph, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    生成层级布局的节点位置
    :param graph: 有向图 (networkx DiGraph)
    :param root: 树的根节点
    :param width: 树的宽度
    :param vert_gap: 层与层之间的垂直间隔
    :param vert_loc: 根节点的垂直位置
    :param xcenter: 根节点的水平中心
    :return: 节点位置字典
    """
    # if not nx.is_tree(graph):
    #     raise TypeError("输入的图不是树结构")

    def _hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=None):
        if pos is None:
            pos = {}
        if parsed is None:
            parsed = set()
        children = list(graph.successors(root))
        if not children:
            pos[root] = (xcenter, vert_loc)
        else:
            if parent is not None:
                children = [c for c in children if c not in parsed]
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap,
                                         xcenter=nextx, pos=pos, parent=root, parsed=parsed)
        pos[root] = (xcenter, vert_loc)
        parsed.add(root)
        return pos

    return _hierarchy_pos(graph, root, width, vert_gap, vert_loc, xcenter)

def visualize_tree_with_ellipsis(node):
    graph = nx.DiGraph()

    def add_edges(node, parent_name=None, action=None, limit_per_level=3):
        """
        限制每层显示前2个和最后1个子节点，中间节点用省略号代替。
        """
        node_name = f"{id(node)}"
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

        label = f"Visits: {node._n_visits}\nQ: {q_value:.2f}\nu: {u_value:.2f}\nP: {p_value:.2f}"
        graph.add_node(node_name, label=label)

        if parent_name is not None and action is not None:
            graph.add_edge(parent_name, node_name, label=str(action))

        children = list(node._children.items())
        num_children = len(children)

        if num_children > limit_per_level + 1:
            # 添加前两个子节点
            for action, child in children[:limit_per_level]:
                add_edges(child, node_name, action)

            # 添加省略号节点
            ellipsis_node_name = f"ellipsis_{id(node)}"
            graph.add_node(ellipsis_node_name, label="...")
            graph.add_edge(node_name, ellipsis_node_name, label="...")

            # 添加最后一个子节点
            action, child = max(children, key=lambda x: x[1]._n_visits)
            add_edges(child, node_name, action)
            action, child = children[-1]
            add_edges(child, node_name, action)
        else:
            # 如果子节点数量不超过限制，则正常添加
            for action, child in children:
                add_edges(child, node_name, action)

    add_edges(node)



    # 使用层级布局
    pos = hierarchical_pos(graph, root=f"{id(node)}")

    # 获取节点和边的标签
    node_labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")

    # 绘制节点和边
    nx.draw(graph, pos, with_labels=True, labels=node_labels, node_size=200,
            node_color="lightblue", font_size=5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=5)
    plt.show()


#visualize_tree_with_ellipsis(root)

def x():
    # 设置页面标题
    st.title('决策过程')

    # 添加一个副标题
    i=0
    st.header(f'mcts树第{i}次更新'.format(i))

    # 添加文本
    st.write('这是一个简单的 Streamlit 示例')
