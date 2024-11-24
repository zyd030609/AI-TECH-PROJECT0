"""使用alphazero打造属于你自己的象棋AI"""

#zyd
#阅读次序
#init
#1.game.py:游戏规则，状态变量，创建了游戏类，用于自我博弈
#2.pytorch_net.py:策略价值网络，=>对某节点生成子节点走子概率+该节点胜率；
#3.mcts.py:蒙特卡洛树搜索，配合了神经网络，还创建了ai玩家类
#4.collect.py:执行自我博弈收集数据
#5.train.py:训练，与collect同时进行
#6.play_with_ai.py:人机对战，创建了真人玩家类
#7.UIplay.py:可交互可视化界面，创建了真人玩家类
#8.config,py:参数配置

#选读
#zip_array.py:数据格式转换，用于数据的存储
#my_redis.py:数据库
#mcts_pure.py:貌似用的不多
#paddle_net.py:

#工作日志
#1.王振鹏（？~11.23）：修改了mcts.py、mcts_pure.py、collect.py、config,py、pytorch_net.py，修改了模型训练中的一些问题
#2.左亦东（11.24）：修改了UIplay.py，解决了人机对战中的3个问题