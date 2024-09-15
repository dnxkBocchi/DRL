import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import tensorflow._api.v2.compat.v1 as tf

"""
Tensorflow Setting
"""
tf.disable_eager_execution()
tf.disable_v2_behavior()
random.seed(6)
np.random.seed(6)
tf.set_random_seed(6)


class baseline_DQN:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.001,
        reward_decay=0.9,
        # 智能体有一定的概率选择随机动作而非当前的最优动作。
        e_greedy=0.99,
        # 每隔多少步更新目标网络的参数。
        replace_target_iter=50,
        memory_size=800,
        batch_size=60,
        e_greedy_increment=0.002,
        # output_graph=False,
    ):
        self.n_actions = n_actions  # if +1: allow to reject jobs
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.01 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0  # total learning step
        self.replay_buffer = deque()  # init experience replay [s, a, r, s_, done]

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection("target_net_params")
        e_params = tf.get_collection("eval_net_params")

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        w_initializer = tf.random_normal_initializer(
            0.0, 0.3, 5
        )  # (mean=0.0, stddev=1.0, seed=None)
        # w_initializer = tf.random_normal_initializer(0., 0.3)  # no seed
        b_initializer = tf.constant_initializer(0.1)
        # 定义隐藏层的神经元数量为 20
        n_l1 = 20  # config of layers

        # ------------------ build evaluate_net ------------------
        # 为输入的状态 s 定义一个占位符，数据类型为 float32，大小为 [None, self.n_features]
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name="s")  # input

        # 定义一个名为 eval_net 的变量域，允许变量共享。
        # 含义: 变量域用来隔离不同网络中的变量，使得评估网络和目标网络的参数可以分开管理。
        with tf.variable_scope("eval_net", reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            # 为当前网络的变量定义集合名称，用于后续参数的管理和保存。
            c_names = ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope("l1"):
                w1 = tf.get_variable(
                    "w1",
                    [self.n_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b1 = tf.get_variable(
                    "b1", [1, n_l1], initializer=b_initializer, collections=c_names
                )
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            # second layer
            with tf.variable_scope("l2"):
                w2 = tf.get_variable(
                    "w2",
                    [n_l1, self.n_actions],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b2 = tf.get_variable(
                    "b2",
                    [1, self.n_actions],
                    initializer=b_initializer,
                    collections=c_names,
                )
                self.q_eval = tf.matmul(l1, w2) + b2

        # --------------------calculate loss---------------------
        self.action_input = tf.placeholder("float", [None, self.n_actions])
        self.q_target = tf.placeholder(
            tf.float32, [None], name="Q_target"
        )  # for calculating loss

        # 计算 Q 估计值。self.q_eval 是当前状态下所有动作的 Q 值，而 self.action_input 是执行的动作的 one-hot 向量。
        q_evaluate = tf.reduce_sum(
            tf.multiply(self.q_eval, self.action_input), reduction_indices=1
        )
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, q_evaluate))
            # 使用 RMSProp 优化器最小化损失函数，并通过梯度下降更新评估网络的参数。
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            print("xxxasdasdasd", self.loss)
            # self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  # better than RMSProp

            # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name="s_"
        )  # input
        with tf.variable_scope("target_net", reuse=tf.AUTO_REUSE):
            # c_names(collections_names) are the collections to store variables
            c_names = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope("l1"):
                w1 = tf.get_variable(
                    "w1",
                    [self.n_features, n_l1],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b1 = tf.get_variable(
                    "b1", [1, n_l1], initializer=b_initializer, collections=c_names
                )
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            # second layer
            with tf.variable_scope("l2"):
                w2 = tf.get_variable(
                    "w2",
                    [n_l1, self.n_actions],
                    initializer=w_initializer,
                    collections=c_names,
                )
                b2 = tf.get_variable(
                    "b2",
                    [1, self.n_actions],
                    initializer=b_initializer,
                    collections=c_names,
                )
                self.q_next = tf.matmul(l1, w2) + b2

            # print('w1:', w1, '  b1:', b1, ' w2:', w2, ' b2:', b2)

    def choose_action(self, state):
        pro = np.random.uniform()
        # epsilon 是探索概率。在动作选择中，有时会选择最优动作（基于当前Q值），有时会随机选择动作（探索）。
        # 如果随机生成的数 pro 小于 epsilon，则选择最优动作；否则，选择一个随机动作。
        if pro < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
            action = np.argmax(actions_value)
            # print('pro: ', pro, ' q-values:', actions_value, '  best_action:', action)
            # print('  best_action:', action)
        else:
            action = np.random.randint(0, self.n_actions)
            # print('pro: ', pro, '  rand_action:', action)
            # print('  rand_action:', action)
        return action

    def choose_best_action(self, state):
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [state]})
        action = np.argmax(actions_value)
        return action

    def store_transition(self, s, a, r, s_):
        one_hot_action = np.zeros(self.n_actions)
        one_hot_action[a] = 1
        self.replay_buffer.append((s, one_hot_action, r, s_))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('-------------target_params_replaced------------------')

        # sample batch memory from all memory: [s, a, r, s_]
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # calculate target q-value batch
        q_next_batch = self.sess.run(self.q_next, feed_dict={self.s_: next_state_batch})
        q_real_batch = []
        for i in range(self.batch_size):
            q_real_batch.append(minibatch[i][2] + self.gamma * np.max(q_next_batch[i]))
        # train eval network
        self.sess.run(
            self._train_op,
            feed_dict={
                self.s: state_batch,
                self.action_input: action_batch,
                self.q_target: q_real_batch,
            },
        )

        # increasing epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max
        # print('epsilon:', self.epsilon)
        self.learn_step_counter += 1


class baselines:
    def __init__(self, n_actions, VMtypes):
        self.n_actions = n_actions
        self.VMtypes = np.array(VMtypes)  # change list to numpy
        # parameters for sensible policy
        self.sensible_updateT = 5
        self.sensible_counterT = 1
        self.sensible_discount = 0.7  # 0.7 is best, 0.5 and 0.6 OK
        self.sensible_W = np.zeros(self.n_actions)
        self.sensible_probs = np.ones(self.n_actions) / self.n_actions
        self.sensible_probsCumsum = self.sensible_probs.cumsum()
        self.sensible_sumDurations = np.zeros(
            (2, self.n_actions)
        )  # row 1: jobNum   row 2: sum duration

    def random_choose_action(self):  # random policy
        action = np.random.randint(self.n_actions)  # [0, n_actions)
        return action

    def RR_choose_action(self, job_count):  # round robin policy
        action = (job_count - 1) % self.n_actions
        return action

    def EITF_choose_action(self, idleTimes):
        action = np.argmin(idleTimes)
        return action

    def BEST_FIT_choose_action(
        self, remaining_CPU, remaining_Memory, remaining_IO, job_attrs
    ):
        reqCPU = job_attrs[3]
        reqMemory = job_attrs[4]
        reqIO = job_attrs[5]
        # Original
        remaining_resources = [
            [
                remaining_CPU[VM_index],
                remaining_Memory[VM_index],
                remaining_IO[VM_index],
            ]
            for VM_index in range(len(remaining_CPU))
        ]
        # sorted CPU -> Memory -> IO
        sorted_index = sorted(
            range(len(remaining_resources)),
            key=lambda x: (
                remaining_resources[x][0],
                remaining_resources[x][1],
                remaining_resources[x][2],
            ),
        )
        index = 0
        while index < len(sorted_index) and (
            remaining_resources[sorted_index[index]][0] < reqCPU
            or remaining_resources[sorted_index[index]][1] < reqMemory
            or remaining_resources[sorted_index[index]][2] < reqIO
        ):
            index += 1
        # There is no condition that satisfies the condition
        # the last one
        action = sorted_index[len(sorted_index) - 1]
        if index != len(remaining_resources):
            action = sorted_index[index]
        return action
