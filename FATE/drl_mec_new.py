# -*- coding:UTF-8 -*-

import tensorflow as tf
import numpy as np
from collections import deque  # 固定长度的队列，可以在队列两端操作
import random
import math
import sympy as sy
import matplotlib.pyplot as plt
import datetime
np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    step_index = 0      # 执行步数。
    state_num = 32       # 状态数。
    action_num = 324      # 动作数。
    OBSERVE = 128     # 训练之前观察多少步。
    BATCH = 64          # 选取的小批量训练样本数。
    FINAL_EPSILON = 0.1    # epsilon 的最小值，当 epsilon 小于该值时，将不在随机选择行为。
    INITIAL_EPSILON = 1     # epsilon 的初始值，epsilon 逐渐减小。
    decay_rate = 0.999
    epsilon = 0               # 探索模式计数。
    learn_step_counter = 0    # 训练步数统计。
    learning_rate = 0.1     # 学习率。
    FINAL_learning_rate = 0.001
    gamma = 0.95               # γ经验折损率。
    memory_size = 2048        # 记忆上限。
    memory_counter = 0        # 当前记忆数。
    replay_memory_store = deque()       # 保存观察到的执行过的行动的存储器，即：曾经经历过的记忆。
    state_queue = deque()
    state_size = 4
    state_list = None                   # 生成一个状态矩阵（6 X 6），每一行代表一个状态。
    action_list = None                  # 生成一个动作矩阵。
    rou = 0.008           # 发射功率权重值
    action_state = [[0, 3, 5, 5], [0, 3, 5, 10], [0, 3, 5, 15], [0, 3, 5, 20], [0, 3, 5, 25], [0, 3, 5, 30], [0, 3, 5, 35], [0, 3, 5, 40], [0, 3, 5, 45],
                    [0, 3, 10, 5], [0, 3, 10, 10], [0, 3, 10, 15], [0, 3, 10, 20], [0, 3, 10, 25], [0, 3, 10, 30], [0, 3, 10, 35], [0, 3, 10, 40], [0, 3, 10, 45],
                    [0, 3, 15, 5], [0, 3, 15, 10], [0, 3, 15, 15], [0, 3, 15, 20], [0, 3, 15, 25], [0, 3, 15, 30], [0, 3, 15, 35], [0, 3, 15, 40], [0, 3, 15, 45],
                    [0, 3, 20, 5], [0, 3, 20, 10], [0, 3, 20, 15], [0, 3, 20, 20], [0, 3, 20, 25], [0, 3, 20, 30], [0, 3, 20, 35], [0, 3, 20, 40], [0, 3, 20, 45],
                    [0, 3, 25, 5], [0, 3, 25, 10], [0, 3, 25, 15], [0, 3, 25, 20], [0, 3, 25, 25], [0, 3, 25, 30], [0, 3, 25, 35], [0, 3, 25, 40], [0, 3, 25, 45],
                    [0, 3, 30, 5], [0, 3, 30, 10], [0, 3, 30, 15], [0, 3, 30, 20], [0, 3, 30, 25], [0, 3, 30, 30], [0, 3, 30, 35], [0, 3, 30, 40], [0, 3, 30, 45],
                    [0, 3, 35, 5], [0, 3, 35, 10], [0, 3, 35, 15], [0, 3, 35, 20], [0, 3, 35, 25], [0, 3, 35, 30], [0, 3, 35, 35], [0, 3, 35, 40], [0, 3, 35, 45],
                    [0, 3, 40, 5], [0, 3, 40, 10], [0, 3, 40, 15], [0, 3, 40, 20], [0, 3, 40, 25], [0, 3, 40, 30], [0, 3, 40, 35], [0, 3, 40, 40], [0, 3, 40, 45],
                    [0, 3, 45, 5], [0, 3, 45, 10], [0, 3, 45, 15], [0, 3, 45, 20], [0, 3, 45, 25], [0, 3, 45, 30], [0, 3, 45, 35], [0, 3, 45, 40], [0, 3, 45, 45],

                    [1, 2, 5, 5], [1, 2, 5, 10], [1, 2, 5, 15], [1, 2, 5, 20], [1, 2, 5, 25], [1, 2, 5, 30], [1, 2, 5, 35], [1, 2, 5, 40], [1, 2, 5, 45],
                    [1, 2, 10, 5], [1, 2, 10, 10], [1, 2, 10, 15], [1, 2, 10, 20], [1, 2, 10, 25], [1, 2, 10, 30], [1, 2, 10, 35], [1, 2, 10, 40], [1, 2, 10, 45],
                    [1, 2, 15, 5], [1, 2, 15, 10], [1, 2, 15, 15], [1, 2, 15, 20], [1, 2, 15, 25], [1, 2, 15, 30], [1, 2, 15, 35], [1, 2, 15, 40], [1, 2, 15, 45],
                    [1, 2, 20, 5], [1, 2, 20, 10], [1, 2, 20, 15], [1, 2, 20, 20], [1, 2, 20, 25], [1, 2, 20, 30], [1, 2, 20, 35], [1, 2, 20, 40], [1, 2, 20, 45],
                    [1, 2, 25, 5], [1, 2, 25, 10], [1, 2, 25, 15], [1, 2, 25, 20], [1, 2, 25, 25], [1, 2, 25, 30], [1, 2, 25, 35], [1, 2, 25, 40], [1, 2, 25, 45],
                    [1, 2, 30, 5], [1, 2, 30, 10], [1, 2, 30, 15], [1, 2, 30, 20], [1, 2, 30, 25], [1, 2, 30, 30], [1, 2, 30, 35], [1, 2, 30, 40], [1, 2, 30, 45],
                    [1, 2, 35, 5], [1, 2, 35, 10], [1, 2, 35, 15], [1, 2, 35, 20], [1, 2, 35, 25], [1, 2, 35, 30], [1, 2, 35, 35], [1, 2, 35, 40], [1, 2, 35, 45],
                    [1, 2, 40, 5], [1, 2, 40, 10], [1, 2, 40, 15], [1, 2, 40, 20], [1, 2, 40, 25], [1, 2, 40, 30], [1, 2, 40, 35], [1, 2, 40, 40], [1, 2, 40, 45],
                    [1, 2, 45, 5], [1, 2, 45, 10], [1, 2, 45, 15], [1, 2, 45, 20], [1, 2, 45, 25], [1, 2, 45, 30], [1, 2, 45, 35], [1, 2, 45, 40], [1, 2, 45, 45],

                    [2, 1, 5, 5], [2, 1, 5, 10], [2, 1, 5, 15], [2, 1, 5, 20], [2, 1, 5, 25], [2, 1, 5, 30], [2, 1, 5, 35], [2, 1, 5, 40], [2, 1, 5, 45],
                    [2, 1, 10, 5], [2, 1, 10, 10], [2, 1, 10, 15], [2, 1, 10, 20], [2, 1, 10, 25], [2, 1, 10, 30], [2, 1, 10, 35], [2, 1, 10, 40], [2, 1, 10, 45],
                    [2, 1, 15, 5], [2, 1, 15, 10], [2, 1, 15, 15], [2, 1, 15, 20], [2, 1, 15, 25], [2, 1, 15, 30], [2, 1, 15, 35], [2, 1, 15, 40], [2, 1, 15, 45],
                    [2, 1, 20, 5], [2, 1, 20, 10], [2, 1, 20, 15], [2, 1, 20, 20], [2, 1, 20, 25], [2, 1, 20, 30], [2, 1, 20, 35], [2, 1, 20, 40], [2, 1, 20, 45],
                    [2, 1, 25, 5], [2, 1, 25, 10], [2, 1, 25, 15], [2, 1, 25, 20], [2, 1, 25, 25], [2, 1, 25, 30], [2, 1, 25, 35], [2, 1, 25, 40], [2, 1, 25, 45],
                    [2, 1, 30, 5], [2, 1, 30, 10], [2, 1, 30, 15], [2, 1, 30, 20], [2, 1, 30, 25], [2, 1, 30, 30], [2, 1, 30, 35], [2, 1, 30, 40], [2, 1, 30, 45],
                    [2, 1, 35, 5], [2, 1, 35, 10], [2, 1, 35, 15], [2, 1, 35, 20], [2, 1, 35, 25], [2, 1, 35, 30], [2, 1, 35, 35], [2, 1, 35, 40], [2, 1, 35, 45],
                    [2, 1, 40, 5], [2, 1, 40, 10], [2, 1, 40, 15], [2, 1, 40, 20], [2, 1, 40, 25], [2, 1, 40, 30], [2, 1, 40, 35], [2, 1, 40, 40], [2, 1, 40, 45],
                    [2, 1, 45, 5], [2, 1, 45, 10], [2, 1, 45, 15], [2, 1, 45, 20], [2, 1, 45, 25], [2, 1, 45, 30], [2, 1, 45, 35], [2, 1, 45, 40], [2, 1, 45, 45],

                    [3, 0, 5, 5], [3, 0, 5, 10], [3, 0, 5, 15], [3, 0, 5, 20], [3, 0, 5, 25], [3, 0, 5, 30], [3, 0, 5, 35], [3, 0, 5, 40], [3, 0, 5, 45],
                    [3, 0, 10, 5], [3, 0, 10, 10], [3, 0, 10, 15], [3, 0, 10, 20], [3, 0, 10, 25], [3, 0, 10, 30], [3, 0, 10, 35], [3, 0, 10, 40], [3, 0, 10, 45],
                    [3, 0, 15, 5], [3, 0, 15, 10], [3, 0, 15, 15], [3, 0, 15, 20], [3, 0, 15, 25], [3, 0, 15, 30], [3, 0, 15, 35], [3, 0, 15, 40], [3, 0, 15, 45],
                    [3, 0, 20, 5], [3, 0, 20, 10], [3, 0, 20, 15], [3, 0, 20, 20], [3, 0, 20, 25], [3, 0, 20, 30], [3, 0, 20, 35], [3, 0, 20, 40], [3, 0, 20, 45],
                    [3, 0, 25, 5], [3, 0, 25, 10], [3, 0, 25, 15], [3, 0, 25, 20], [3, 0, 25, 25], [3, 0, 25, 30], [3, 0, 25, 35], [3, 0, 25, 40], [3, 0, 25, 45],
                    [3, 0, 30, 5], [3, 0, 30, 10], [3, 0, 30, 15], [3, 0, 30, 20], [3, 0, 30, 25], [3, 0, 30, 30], [3, 0, 30, 35], [3, 0, 30, 40], [3, 0, 30, 45],
                    [3, 0, 35, 5], [3, 0, 35, 10], [3, 0, 35, 15], [3, 0, 35, 20], [3, 0, 35, 25], [3, 0, 35, 30], [3, 0, 35, 35], [3, 0, 35, 40], [3, 0, 35, 45],
                    [3, 0, 40, 5], [3, 0, 40, 10], [3, 0, 40, 15], [3, 0, 40, 20], [3, 0, 40, 25], [3, 0, 40, 30], [3, 0, 40, 35], [3, 0, 40, 40], [3, 0, 40, 45],
                    [3, 0, 45, 5], [3, 0, 45, 10], [3, 0, 45, 15], [3, 0, 45, 20], [3, 0, 45, 25], [3, 0, 45, 30], [3, 0, 45, 35], [3, 0, 45, 40], [3, 0, 45, 45]
                    ]
    # action_state = [[0, 3],[1, 2], [2, 1], [3, 0]]

    # MEC 模型参数
    cpu1 = None
    cpu2 = None
    ts = 4.5 * 10 ** (-6)
    Tf = 600
    mUL = 200
    L = 2640
    f0 = 3 * 10 ** 9
    # detSNR = 25
    rouDL = 0.7
    ErrorP = 10 ** (-4)
    alpha = 3
    beta = 0.25
    task_arrival_probability = 0.4
    random_num = None
    wait_counter1 = 0
    wait_counter2 = 0
    step_counter_zero1 = 0
    step_counter_one1 = 0
    step_counter_zero2 = 0
    step_counter_one2 = 0

    DQNSucess = 0
    DQNDrop = 0

    buffer1 = deque()
    buffer2 = deque()
    hDL_buffer1 = deque()
    hDL_buffer2 = deque()
    wait_time = 3
    transform_array1 = None
    transform_array2 = None
    last_state = None
    task_success_ratio = []
    loss_state = []
    power_state = []
    power_buffer = []
    power_sum = 0
    wight = [
        0.308441115765020141547470834677860695628728886538337442115557180200185562029325570030406854605797422692709124967333269311325976,
        0.40111992915527355151578030991281951479548361696211301757496775516766091500023676907078271471115370136487330141071993586230177,
        0.2180682876118094215886485234746467267427785384121889405665039813207866459828519234900012632360320998206316510456964215584149,
        0.0620874560986777473929021293135179536959090656838020923686767603039238664629902769437184001766749562167627176007775000692659,
        0.0095015169751811005538390721941719912258624504015797533646318970848680322451340750784240877901570876971156524042858250386085,
        0.00075300838858753877545596435367566390179203914014362886506164014367761532732553587715637849496678837150424361526704126458678,
        0.000028259233495995655674225638268500212828033164744374677117086795259322508664275187881955981278223738401217175272290547273269,
        4.2493139849626863725865766597471235464810801986441570081254604829021791344429695503912683040809284918604347723905036055947e-7,
        1.8395648239796307809215352243559382479826127765906584095132767065586955845890346516805867002792829341205430551248512322204e-9,
        9.9118272196090085583775472832447360645810946112447692431304366686254543563865419652261172586997161662742217301111109553476e-13
    ]
    node = [
        0.13779347054049243083077250565271118810799168074578329416336651137344596447646208656254375241734116529355248965683739196203130605,
        0.72945454950317049816037312167607878107607273331224924600781104880626782307305157204652499178629631939583813469724208401283647042,
        1.8083429017403160482329200757506088332830602823714481516932097683732967664043428239153383606124012239270279810042946898231255533,
        3.4014336978548995144825322214083906792731566142034732294267661021823269228666946410395095870847780590663426380062014239602311845,
        5.5524961400638036324175584868687628579740642873178138284904788336252636084290435460571874259935064862774598817665863631336209552,
        8.3301527467644967002387671972745221827094389720309893447446294283573954320532476951665824491858289576075525949818506146692432342,
        11.843785837900065564918538919141613985802816909465335873339591495276220150697108232717212527603241705173109616546015446373017787,
        16.279257831378102099532653935833622335255995603033059077788392165002827350932597940225841507596519165747859331720868398463246006,
        21.996585811980761951277090195594493976806732340001887788241503511693106375798998823067686533346980361167369020416848451979346621,
        29.920697012273891559908793340799195179710670577517960166104251135309849605268452639201572864373106556343888311203255135623300882
    ]
    transitionMatrix = [[0.9, 0.1, 0.9, 0.1],
                        [0.8, 0.2, 0.8, 0.2],
                        [0.7, 0.3, 0.7, 0.3],
                        [0.6, 0.4, 0.6, 0.4],
                        # [0.8, 0.2, 0.3, 0.7],
                        [0.9, 0.1, 0.1, 0.9],
                        [0.4, 0.6, 0.4, 0.6],
                        # [0.7, 0.3, 0.2, 0.8],
                        [0.3, 0.7, 0.3, 0.7],
                        [0.2, 0.8, 0.2, 0.8],
                        [0.1, 0.9, 0.1, 0.9],
                        [0, 1.0, 0, 1.0]
                        ]
    markov_state = [0, 1]
    markov_next_state = [0, 0]
    sum1 = 0
    sum2 = 0

    # q_eval 网络。
    q_eval_input = None
    action_input = None
    q_target = None
    q_eval = None
    predict = None
    loss = None
    train_op = None
    cost_his = None
    reward_action = None
    ep_reward = 0
    reward_view = []

    session = None    # tensorflow 会话

    def __init__(self, learning_rate=0.1, gamma=0.95, memory_size=2048):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size
        self.action_list = np.identity(self.action_num)        # 初始化成一个动作矩阵。
        self.create_network()        # 创建神经网络。
        self.session = tf.InteractiveSession()                 # 初始化 tensorflow 会话。
        self.session.run(tf.global_variables_initializer())        # 初始化 tensorflow 参数。
        self.cost_his = []           # 记录所有 loss 变化。
        self.loss_state = []
        self.power_state = []

    def create_network(self):
        """
        创建神经网络。
        :return:
        """
        self.q_eval_input = tf.placeholder(shape=[None, self.state_num], dtype=tf.float32)   # 占位符
        self.action_input = tf.placeholder(shape=[None, self.action_num], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)

        neuro_layer_1 = 64     # 隐层神经数目
        w1 = tf.Variable(tf.random_normal([self.state_num, neuro_layer_1]))
        b1 = tf.Variable(tf.zeros([1, neuro_layer_1]) + 0.1)
        l1 = tf.nn.sigmoid(tf.matmul(self.q_eval_input, w1) + b1)

        neuro_layer_2 = 64     # 隐层神经数目
        w2 = tf.Variable(tf.random_normal([neuro_layer_1, neuro_layer_2]))
        b2 = tf.Variable(tf.zeros([1, neuro_layer_2]) + 0.1)
        l2 = tf.nn.sigmoid(tf.matmul(l1, w2) + b2)

        # neuro_layer_3 = 128     # 隐层神经数目
        # w3 = tf.Variable(tf.random_normal([neuro_layer_2, neuro_layer_3]))
        # b3 = tf.Variable(tf.zeros([1, neuro_layer_3]) + 0.1)
        # l3 = tf.nn.tanh(tf.matmul(l2, w3) + b3)
        #
        # neuro_layer_4 = 128     # 隐层神经数目
        # w4 = tf.Variable(tf.random_normal([neuro_layer_3, neuro_layer_4]))
        # b4 = tf.Variable(tf.zeros([1, neuro_layer_4]) + 0.1)
        # l4 = tf.nn.tanh(tf.matmul(l3, w4) + b4)

        neuro_layer_3 = 64     # 隐层神经数目
        w3 = tf.Variable(tf.random_normal([neuro_layer_2, neuro_layer_3]))
        b3 = tf.Variable(tf.zeros([1, neuro_layer_3]) + 0.1)
        l3 = tf.nn.sigmoid(tf.matmul(l2, w3) + b3)

        neuro_layer_4 = 64     # 隐层神经数目
        w4 = tf.Variable(tf.random_normal([neuro_layer_3, neuro_layer_4]))
        b4 = tf.Variable(tf.zeros([1, neuro_layer_4]) + 0.1)
        l4 = tf.nn.sigmoid(tf.matmul(l3, w4) + b4)

        w5 = tf.Variable(tf.random_normal([neuro_layer_4, self.action_num]))
        b5 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
        self.q_eval = tf.matmul(l4, w5) + b5

        # 取出当前动作的得分。
        self.reward_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square((self.q_target - self.reward_action)))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        if self.learning_rate > self.FINAL_learning_rate:
            self.learning_rate = self.learning_rate * self.decay_rate

        self.predict = tf.argmax(self.q_eval, 1)

    def select_action(self, state_index):
        """
        根据策略选择动作。
        :param state_index: 当前状态。
        :return:
        """
        current_state = state_index
        current_state = np.reshape(current_state, (1, 32))
        if np.random.uniform() < self.epsilon:
            current_action_index = np.random.randint(0, self.action_num)
        else:
            actions_value = self.session.run(self.q_eval, feed_dict={self.q_eval_input: current_state})
            action = np.argmax(actions_value)
            current_action_index = action

        # 开始训练后，在 epsilon 小于一定的值之前，将逐步减小 epsilon。
        if self.epsilon > self.FINAL_EPSILON:
            self.epsilon = self.epsilon * self.decay_rate

        return current_action_index

    def save_store(self, current_state_index, current_action_index, current_reward, next_state_index):
        """
        保存记忆。
        :param current_state_index: 当前状态 index。
        :param current_action_index: 动作 index。
        :param current_reward: 奖励。
        :param next_state_index: 下一个状态 index。
        :return:
        """
        current_state = current_state_index
        current_state = np.reshape(current_state, (1, 32))
        current_action = self.action_list[current_action_index:current_action_index + 1]
        next_state = next_state_index
        next_state = np.reshape(next_state, (1, 32))
        # 记忆动作(当前状态， 当前执行的动作， 当前动作的得分，下一个状态)。
        self.replay_memory_store.append((
            current_state,
            current_action,
            current_reward,
            next_state))

        # 如果超过记忆的容量，则将最久远的记忆移除。
        if len(self.replay_memory_store) > self.memory_size:
            self.replay_memory_store.popleft()

        self.memory_counter += 1

    def step(self, state, action):
        """
        执行动作。
        :param state: 当前状态。
        :param action: 执行的动作。
        :return:
        """
        # starttime = datetime.datetime.now()
        # print(starttime)

        state = np.reshape(state, (1, 8))
        # t = sy.Symbol('t')
        hdl1_buffer = self.hDL_buffer1.copy()
        hdl2_buffer = self.hDL_buffer2.copy()
        dul1 = 500 * state[0][0]
        dul2 = 500 * state[0][1]
        ddl1 = self.beta * dul1
        ddl2 = self.beta * dul2
        w1 = state[0][2]
        w2 = state[0][3]
        q1 = state[0][4]
        q2 = state[0][5]

        self.cpu1 = self.action_state[action][0]
        self.cpu2 = self.action_state[action][1]
        SNR1 = self.action_state[action][2]
        self.power_state.append(SNR1)
        SNR2 = self.action_state[action][3]
        self.power_state.append(SNR2)
        # SNR1 = 25
        # self.power_state.append(SNR1)
        # SNR2 = 25
        # self.power_state.append(SNR2)

        hdl1 = self.rouDL * hdl1_buffer[0] + \
               math.sqrt(1 - self.rouDL ** 2) * math.sqrt(1 / 2) * complex(np.random.normal(), np.random.normal())
        hdl2 = self.rouDL * hdl2_buffer[0] + \
               math.sqrt(1 - self.rouDL ** 2) * math.sqrt(1 / 2) * complex(np.random.normal(), np.random.normal())
        snr1 = SNR1 * (abs(hdl1)) ** 2
        snr2 = SNR2 * (abs(hdl2)) ** 2
        self.hDL_buffer1.append(hdl1)
        self.hDL_buffer2.append(hdl2)
        if len(self.hDL_buffer1) > 1:
            self.hDL_buffer1.popleft()
        if len(self.hDL_buffer2) > 1:
            self.hDL_buffer2.popleft()

        if (len(self.buffer1) == 0) and (len(self.buffer2) == 0):
            self.wait_counter1 = 0
            self.wait_counter2 = 0
            reward1 = 0
            reward2 = 0
        if (len(self.buffer1) == 0) and (len(self.buffer2) != 0):
            if self.cpu2 == 0:
                w2 += 1
                reward2 = 0
                reward1 = 0
            else:
                if len(self.buffer2) == 2:
                    if self.step_counter_one2 < self.step_counter_zero2:
                        w2 = w2
                    if self.step_counter_zero2 < self.step_counter_one2:
                        w2 = w2 - self.wait_counter2
                        self.wait_counter2 = 0
                else:
                    w2 = w2 - self.wait_counter2
                    if w2 < 0:
                         w2 = 0
                    self.wait_counter2 = 0
                q2 = q2 - 1
                self.buffer2.popleft()
                mc2 = math.ceil((dul2 * self.L) / (self.cpu2 * self.f0 * self.ts))
                mdl2 = self.Tf - mc2 - self.mUL
                x2 = (math.log((1 + snr2), 2) - ddl2 / mdl2) / \
                        (math.log(math.exp(1), 2) * math.sqrt((1 - 1 / ((1 + snr2) ** 2)) / mdl2))
                def f(x):
                    if x2 <= 0:
                        return 1 - (1 / (2 * sy.sqrt((x + (x2 ** 2) / 2) * sy.pi))) * sy.exp(-(x2 ** 2) / 2)
                    else:
                        return (1 / (2 * sy.sqrt((x + (x2 ** 2) / 2) * sy.pi))) * sy.exp(-(x2 ** 2) / 2)
                for i in range(10):
                    self.sum2 += self.wight[i] * f(self.node[i])
                errorp2 = float(self.sum2)
                self.sum2 = 0
                reward1 = 0
                if errorp2 > self.ErrorP:
                    reward2 = -1 - self.rou * SNR2
                    self.DQNDrop = self.DQNDrop + 1
                else:
                    reward2 = 1 - self.rou * SNR2
                    self.DQNSucess = self.DQNSucess + 1
            self.wait_counter1 = 0
        if (len(self.buffer1) != 0) and (len(self.buffer2) == 0):
            if self.cpu1 == 0:
                w1 += 1
                reward1 = 0
                reward2 = 0
            else:
                if len(self.buffer1) == 2:
                    if self.step_counter_one1 < self.step_counter_zero1:
                        w1 = w1
                    if self.step_counter_zero1 < self.step_counter_one1:
                        w1 = w1 - self.wait_counter1
                        self.wait_counter1 = 0
                else:
                    w1 = w1 - self.wait_counter1
                    if w1 < 0:
                        w1 = 0
                    self.wait_counter1 = 0
                q1 = q1 - 1
                self.buffer1.popleft()
                mc1 = math.ceil((dul1 * self.L) / (self.cpu1 * self.f0 * self.ts))
                mdl1 = self.Tf - mc1 - self.mUL
                x1 = (math.log((1 + snr1), 2) - ddl1 / mdl1) / \
                     (math.log(math.exp(1), 2) * math.sqrt((1 - 1 / ((1 + snr1) ** 2)) / mdl1))
                def g(t):
                    if x1 <= 0:
                        return 1 - (1 / (2 * sy.sqrt((t + (x1 ** 2) / 2) * sy.pi))) * sy.exp(-(x1 ** 2) / 2)
                    else:
                        return (1 / (2 * sy.sqrt((t + (x1 ** 2) / 2) * sy.pi))) * sy.exp(-(x1 ** 2) / 2)
                for i in range(10):
                    self.sum1 += self.wight[i] * g(self.node[i])
                errorp1 = float(self.sum1)
                self.sum1 = 0
                if errorp1 > self.ErrorP:
                    reward1 = -1 - self.rou * SNR1
                    self.DQNDrop = self.DQNDrop + 1
                else:
                    reward1 = 1 - self.rou * SNR1
                    self.DQNSucess = self.DQNSucess + 1
                reward2 = 0
            self.wait_counter2 = 0
        if (len(self.buffer1) != 0) and (len(self.buffer2) != 0):
            if self.cpu1 == 0:
                w1 += 1
                reward1 = 0
            else:
                if len(self.buffer1) == 2:
                    if self.step_counter_one1 < self.step_counter_zero1:
                        w1 = w1
                    if self.step_counter_zero1 < self.step_counter_one1:
                        w1 = w1 - self.wait_counter1
                        self.wait_counter1 = 0
                else:
                    w1 = w1 - self.wait_counter1
                    if w1 < 0:
                        w1 = 0
                    self.wait_counter1 = 0
                q1 = q1 - 1
                self.buffer1.popleft()
                mc1 = math.ceil((dul1 * self.L) / (self.cpu1 * self.f0 * self.ts))
                mdl1 = self.Tf - mc1 - self.mUL
                x1 = (math.log((1 + snr1), 2) - ddl1 / mdl1) / \
                     (math.log(math.exp(1), 2) * math.sqrt((1 - 1 / ((1 + snr1) ** 2)) / mdl1))
                def g(t):
                    if x1 <= 0:
                        return 1 - (1 / (2 * sy.sqrt((t + (x1 ** 2) / 2) * sy.pi))) * sy.exp(-(x1 ** 2) / 2)
                    else:
                        return (1 / (2 * sy.sqrt((t + (x1 ** 2) / 2) * sy.pi))) * sy.exp(-(x1 ** 2) / 2)
                for i in range(10):
                    self.sum1 += self.wight[i] * g(self.node[i])
                errorp1 = float(self.sum1)
                self.sum1 = 0
                if errorp1 > self.ErrorP:
                    reward1 = -1 - self.rou * SNR1
                    self.DQNDrop = self.DQNDrop + 1
                else:
                    reward1 = 1 - self.rou * SNR1
                    self.DQNSucess = self.DQNSucess + 1
            if self.cpu2 == 0:
                w2 += 1
                reward2 = 0
            else:
                if len(self.buffer2) == 2:
                    if self.step_counter_one2 < self.step_counter_zero2:
                        w2 = w2
                    if self.step_counter_zero2 < self.step_counter_one2:
                        w2 = w2 - self.wait_counter2
                        self.wait_counter2 = 0
                else:
                    w2 = w2 - self.wait_counter2
                    if w2 < 0:
                         w2 = 0
                    self.wait_counter2 = 0
                q2 = q2 - 1
                self.buffer2.popleft()
                mc2 = math.ceil((dul2 * self.L) / (self.cpu2 * self.f0 * self.ts))
                mdl2 = self.Tf - mc2 - self.mUL
                x2 = (math.log((1 + snr2), 2) - ddl2 / mdl2) / \
                        (math.log(math.exp(1), 2) * math.sqrt((1 - 1 / ((1 + snr2) ** 2)) / mdl2))
                def f(x):
                    if x2 <= 0:
                        return 1 - (1 / (2 * sy.sqrt((x + (x2 ** 2) / 2) * sy.pi))) * sy.exp(-(x2 ** 2) / 2)
                    else:
                        return (1 / (2 * sy.sqrt((x + (x2 ** 2) / 2) * sy.pi))) * sy.exp(-(x2 ** 2) / 2)
                for i in range(10):
                    self.sum2 += self.wight[i] * f(self.node[i])
                errorp2 = float(self.sum2)
                self.sum2 = 0
                if errorp2 > self.ErrorP:
                    reward2 = -1 - self.rou * SNR2
                    self.DQNDrop = self.DQNDrop + 1
                else:
                    reward2 = 1 - self.rou * SNR2
                    self.DQNSucess = self.DQNSucess + 1

        if w1 >= self.wait_time:
            if len(self.buffer1) == 2:
                if self.step_counter_one1 < self.step_counter_zero1:
                    w1 = w1 - 1
                if self.step_counter_zero1 < self.step_counter_one1:
                    w1 = w1 - 1 - self.wait_counter1
                    self.wait_counter1 = 0
            else:
                w1 = w1 - 1 - self.wait_counter1
                if w1 < 0:
                    w1 = 0
                self.wait_counter1 = 0
            self.buffer1.popleft()
            q1 = q1 - 1
            self.DQNDrop = self.DQNDrop + 1
            reward1 = -1.5
        if w2 >= self.wait_time:
            if len(self.buffer2) == 2:
                if self.step_counter_one2 < self.step_counter_zero2:
                    w2 = w2 - 1
                if self.step_counter_zero2 < self.step_counter_one2:
                    w2 = w2 - 1 - self.wait_counter2
                    self.wait_counter2 = 0
            else:
                w2 = w2 - 1 - self.wait_counter2
                if w2 < 0:
                    w2 = 0
                self.wait_counter2 = 0
            self.buffer2.popleft()
            q2 = q2 - 1
            self.DQNDrop = self.DQNDrop + 1
            reward2 = -1.5

        if len(self.buffer1) == 0:
            w1 = 0
        if len(self.buffer2) == 0:
            w2 = 0

        # self.random_num = np.random.rand(2)
        # if (self.random_num[0] < self.task_arrival_probability) and\
        #         (self.random_num[1] < self.task_arrival_probability):
        #     self.buffer1.append(np.random.randint(1, 5))
        #     self.step_counter_one1 = self.step_index
        #     self.buffer2.append(np.random.randint(1, 5))
        #     self.step_counter_one2 = self.step_index
        #     q1 += 1
        #     q2 += 1
        # if (self.random_num[0] < self.task_arrival_probability) and\
        #         (self.random_num[1] >= self.task_arrival_probability):
        #     self.buffer1.append(np.random.randint(1, 5))
        #     self.step_counter_one1 = self.step_index
        #     q1 += 1
        #     self.step_counter_zero2 = self.step_index
        #     self.wait_counter2 += 1
        # if (self.random_num[0] >= self.task_arrival_probability) and\
        #         (self.random_num[1] < self.task_arrival_probability):
        #     self.buffer2.append(np.random.randint(1, 5))
        #     self.step_counter_one2 = self.step_index
        #     q2 += 1
        #     self.step_counter_zero1 = self.step_index
        #     self.wait_counter1 += 1
        # if (self.random_num[0] >= self.task_arrival_probability) and \
        #         (self.random_num[1] >= self.task_arrival_probability):
        #     self.step_counter_zero1 = self.step_index
        #     self.step_counter_zero2 = self.step_index
        #     self.wait_counter1 += 1
        #     self.wait_counter2 += 1

        if (self.markov_state[0] == 1) and (self.markov_state[1] == 1):
            self.buffer1.append(np.random.randint(1, 5))
            self.step_counter_one1 = self.step_index
            self.buffer2.append(np.random.randint(1, 5))
            self.step_counter_one2 = self.step_index
            q1 += 1
            q2 += 1
        if (self.markov_state[0] == 1) and (self.markov_state[1] == 0):
            self.buffer1.append(np.random.randint(1, 5))
            self.step_counter_one1 = self.step_index
            q1 += 1
            self.step_counter_zero2 = self.step_index
            self.wait_counter2 += 1
        if (self.markov_state[0] == 0) and (self.markov_state[1] == 1):
            self.buffer2.append(np.random.randint(1, 5))
            self.step_counter_one2 = self.step_index
            q2 += 1
            self.step_counter_zero1 = self.step_index
            self.wait_counter1 += 1
        if (self.markov_state[0] == 0) and (self.markov_state[1] == 0):
            self.step_counter_zero1 = self.step_index
            self.step_counter_zero2 = self.step_index
            self.wait_counter1 += 1
            self.wait_counter2 += 1

        if self.markov_state[0] == 0:
            if np.random.rand() <= self.transitionMatrix[2][0]:
                self.markov_next_state[0] = 0
            else:
                self.markov_next_state[0] = 1
        elif self.markov_state[0] == 1:
            if np.random.rand() <= self.transitionMatrix[2][2]:
                self.markov_next_state[0] = 0
            else:
                self.markov_next_state[0] = 1
        if self.markov_state[1] == 0:
            if np.random.rand() <= self.transitionMatrix[2][0]:
                self.markov_next_state[1] = 0
            else:
                self.markov_next_state[1] = 1
        elif self.markov_state[1] == 1:
            if np.random.rand() <= self.transitionMatrix[2][2]:
                self.markov_next_state[1] = 0
            else:
                self.markov_next_state[1] = 1
        self.markov_state = self.markov_next_state

        # snr1 = snr1 / SNR1
        # snr2 = snr2 / SNR2
        snr1 = snr1 / 25
        snr2 = snr2 / 25
        q1 = len(self.buffer1)
        q2 = len(self.buffer2)

        if (len(self.buffer1) == 0) and (len(self.buffer2) == 0):
            next_state = [0, 0, w1, w2, q1, q2, snr1, snr2]
        if (len(self.buffer1) == 0) and (len(self.buffer2) > 0):
            self.transform_array2 = self.buffer2.copy()
            next_state = [0, self.transform_array2[0], w1, w2, q1, q2, snr1, snr2]
        if (len(self.buffer1) > 0) and (len(self.buffer2) == 0):
            self.transform_array1 = self.buffer1.copy()
            next_state = [self.transform_array1[0], 0, w1, w2, q1, q2, snr1, snr2]
        if (len(self.buffer1) > 0) and (len(self.buffer2) > 0):
            self.transform_array1 = self.buffer1.copy()
            self.transform_array2 = self.buffer2.copy()
            next_state = [self.transform_array1[0], self.transform_array2[0], w1, w2, q1, q2, snr1, snr2]

        # endtime = datetime.datetime.now()
        # print(endtime)
        # print(endtime - starttime)

        reward = reward1 + reward2
        return next_state, reward

    def experience_replay(self):
        """
        记忆回放。
        :return:
        """
        # 随机选择一小批记忆样本。
        batch = self.BATCH if self.memory_counter > self.BATCH else self.memory_counter
        minibatch = random.sample(self.replay_memory_store, batch)

        batch_state = None
        batch_action = None
        batch_reward = None
        batch_next_state = None

        for index in range(len(minibatch)):
            if batch_state is None:
                batch_state = minibatch[index][0]
            elif batch_state is not None:
                batch_state = np.vstack((batch_state, minibatch[index][0]))

            if batch_action is None:
                batch_action = minibatch[index][1]
            elif batch_action is not None:
                batch_action = np.vstack((batch_action, minibatch[index][1]))

            if batch_reward is None:
                batch_reward = minibatch[index][2]
            elif batch_reward is not None:
                batch_reward = np.vstack((batch_reward, minibatch[index][2]))

            if batch_next_state is None:
                batch_next_state = minibatch[index][3]
            elif batch_next_state is not None:
                batch_next_state = np.vstack((batch_next_state, minibatch[index][3]))

        # q_next：下一个状态的 Q 值。
        q_next = self.session.run([self.q_eval], feed_dict={self.q_eval_input: batch_next_state})
        q_target = []
        for i in range(len(minibatch)):
            # 当前即时得分。
            current_reward = batch_reward[i][0]
            # 更新 Q 值。
            q_value = current_reward + self.gamma * np.max(q_next[0][i])
            q_target.append(q_value)

        # batch_state = minibatch[0][0]
        # batch_action = minibatch[0][1]
        # batch_reward = minibatch[0][2]
        # batch_next_state = minibatch[0][3]

        # q_next：下一个状态的 Q 值。
        # q_next = self.session.run([self.q_eval], feed_dict={self.q_eval_input: batch_next_state})
        # q_target = batch_reward + self.gamma * np.max(q_next[0])
        # q_target = np.reshape(q_target, (1,))
        _, cost, reward = self.session.run([self.train_op, self.loss, self.reward_action],
                                           feed_dict={self.q_eval_input: batch_state,
                                                      self.action_input: batch_action,
                                                      self.q_target: q_target})

        self.cost_his.append(cost)
        if self.step_index % 100 == 0:
             print("loss:", cost)
             self.loss_state.append(cost)
        self.learn_step_counter += 1

        if self.learning_rate > self.FINAL_learning_rate:
            self.learning_rate = self.learning_rate * 0.999

    def train(self, state, episode):
        """
        训练。
        :return:
        """
        current_state = state
        while True:
            # 选择动作。
            action = self.select_action(current_state)
            # 执行动作，得到：下一个状态，执行动作的得分，是否结束。
            time_slot_next_state, reward = self.step(current_state[3], action)
            self.ep_reward += reward
            # 状态记忆存储
            self.state_queue.append(time_slot_next_state)
            # 如果超过记忆的容量，则将最久远的记忆移除。
            if len(self.state_queue) > self.state_size:
                self.state_queue.popleft()
            next_state = self.state_queue.copy()
            # 保存记忆。
            self.save_store(current_state, action, reward, next_state)
            # 先观察一段时间累积足够的记忆在进行训练。
            if (self.step_index > self.OBSERVE) or (episode >= 1):
                self.experience_replay()
            if self.step_index > 500:
                self.last_state = next_state
                print(self.ep_reward)
                self.reward_view.append(self.ep_reward)
                break
            current_state = next_state
            self.step_index += 1

    def pay(self):
        """
        运行并测试。
        :return:
        """
        # starttime = datetime.datetime.now()
        # 可视化
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_data = np.linspace(1, 150, 150)
        # task_success_ratio = []

        self.epsilon = self.INITIAL_EPSILON
        hdl1 = 1
        hdl2 = 1
        self.hDL_buffer1.append(hdl1)
        self.hDL_buffer2.append(hdl2)
        # self.random_num = np.random.rand(2)
        # if (self.random_num[0] < self.task_arrival_probability) and (self.random_num[1] < self.task_arrival_probability):
        #     time_slot_state = np.array([[np.random.randint(1, 5), np.random.randint(1, 5), 0, 0, 1, 1, 1, 1]])
        #     self.buffer1.append(time_slot_state[0][0])
        #     self.buffer2.append(time_slot_state[0][1])
        # if (self.random_num[0] < self.task_arrival_probability) and (self.random_num[1] >= self.task_arrival_probability):
        #     time_slot_state = np.array([[np.random.randint(1, 5), 0, 0, 0, 1, 0, 1, 1]])
        #     self.buffer1.append(time_slot_state[0][0])
        # if (self.random_num[0] >= self.task_arrival_probability) and (self.random_num[1] < self.task_arrival_probability):
        #     time_slot_state = np.array([[0, np.random.randint(1, 5), 0, 0, 0, 1, 1, 1]])
        #     self.buffer2.append(time_slot_state[0][1])
        # if (self.random_num[0] >= self.task_arrival_probability) and (self.random_num[1] >= self.task_arrival_probability):
        #     time_slot_state = np.array([[0, 0, 0, 0, 0, 0, 1, 1]])

        if (self.markov_state[0] == 1) and (self.markov_state[1] == 1):
            time_slot_state = np.array([[np.random.randint(1, 5), np.random.randint(1, 5), 0, 0, 1, 1, 1, 1]])
            self.buffer1.append(time_slot_state[0][0])
            self.buffer2.append(time_slot_state[0][1])
        if (self.markov_state[0] == 1) and (self.markov_state[1] == 0):
            time_slot_state = np.array([[np.random.randint(1, 5), 0, 0, 0, 1, 0, 1, 1]])
            self.buffer1.append(time_slot_state[0][0])
        if (self.markov_state[0] == 0) and (self.markov_state[1] == 1):
            time_slot_state = np.array([[0, np.random.randint(1, 5), 0, 0, 0, 1, 1, 1]])
            self.buffer2.append(time_slot_state[0][1])
        if (self.markov_state[0] == 0) and (self.markov_state[1] == 0):
            time_slot_state = np.array([[0, 0, 0, 0, 0, 0, 1, 1]])

        if self.markov_state[0] == 0:
            if np.random.rand() <= self.transitionMatrix[2][0]:
                self.markov_next_state[0] = 0
            else:
                self.markov_next_state[0] = 1
        elif self.markov_state[0] == 1:
            if np.random.rand() <= self.transitionMatrix[2][2]:
                self.markov_next_state[0] = 0
            else:
                self.markov_next_state[0] = 1
        if self.markov_state[1] == 0:
            if np.random.rand() <= self.transitionMatrix[2][0]:
                self.markov_next_state[1] = 0
            else:
                self.markov_next_state[1] = 1
        elif self.markov_state[1] == 1:
            if np.random.rand() <= self.transitionMatrix[2][2]:
                self.markov_next_state[1] = 0
            else:
                self.markov_next_state[1] = 1
        self.markov_state = self.markov_next_state

        time_slot_state = np.reshape(time_slot_state, (1, 8))
        for i in range(4):
            current_action = np.random.randint(0, self.action_num)
            time_slot_next_state, reward = self.step(time_slot_state, current_action)
            self.state_queue.append(time_slot_next_state)
            time_slot_state = time_slot_next_state
        self.step_index += 1
        current_state = self.state_queue.copy()
        for index in range(150):
            self.ep_reward = 0
            self.train(current_state, index)
            current_state = self.last_state
            self.DQNDrop = 0
            self.DQNSucess = 0
            self.step_index = 0
            total_power = 0
            self.power_state.clear()
            while True:
                current_state_ = np.reshape(current_state, (1, 32))
                actions_value = self.session.run(self.q_eval, feed_dict={self.q_eval_input: current_state_})
                action = np.argmax(actions_value)

                # print(current_state[3])
                # print(actions_value)
                # print(action)

                time_slot_next_state, reward = self.step(current_state[3], action)
                # 状态记忆存储
                self.state_queue.append(time_slot_next_state)
                # 如果超过记忆的容量，则将最久远的记忆移除。
                if len(self.state_queue) > self.state_size:
                    self.state_queue.popleft()
                next_state = self.state_queue.copy()
                if self.step_index > 2000:
                    success_ratio = self.DQNSucess / (self.DQNSucess + self.DQNDrop)
                    self.task_success_ratio.append(success_ratio)
                    for i in range(len(self.power_state)):
                        total_power += self.power_state[i]
                    total_power = total_power / len(self.power_state)
                    print("episode:", index+1)
                    print(self.DQNSucess)
                    print(self.DQNDrop)
                    print(success_ratio)
                    print(self.learning_rate)
                    print(total_power)
                    self.power_buffer.append(total_power)
                    break
                current_state = next_state
                self.step_index += 1
            current_state = self.last_state
            self.step_index = 0
            self.DQNDrop = 0
            self.DQNSucess = 0

        ax.plot(x_data, self.task_success_ratio)
        plt.ylim(0, 1)
        plt.show()

        x = np.arange(0, len(self.reward_view))
        y = self.reward_view
        plt.plot(x, y)
        plt.show()

        x1 = np.arange(0, len(self.loss_state))
        y1 = self.loss_state
        plt.plot(x1, y1)
        plt.show()

        su_average = 0
        for i in range(50):
            su_average += self.task_success_ratio[len(self.task_success_ratio) - i - 1]
        print(float(su_average / 50))

        # average = 0
        # for i in range(80000):
        #     average += self.power_state[len(self.power_state) - i - 1]
        # print(float(average / 80000))

        for i in range(50):
            self.power_sum += self.power_buffer[len(self.power_buffer) - i - 1]
        print(self.power_sum / 50)


        # endtime = datetime.datetime.now()
        # print(endtime - starttime)

if __name__ == "__main__":
    q_network = DeepQNetwork()
    q_network.pay()