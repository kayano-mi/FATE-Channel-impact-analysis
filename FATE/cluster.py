from __future__ import absolute_import, division, print_function
import rayleigh
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import collections
import numpy as np
np.set_printoptions(threshold = 1e6)#设置打印数量的阈值
from six.moves import range
import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()


NUM_EXAMPLES_PER_USER = 5000  # max = 5421
BATCH_SIZE = 1000  #小于num且最好为num的整数倍

# 入口数据处理
def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, NUM_EXAMPLES_PER_USER,BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        # print(batch_samples)
        output_sequence.append({
            'x':
            np.array(
                [source[0][i].flatten() / 255.0 for i in batch_samples], #灰度归一化
                dtype=np.float32),
            'y':
            np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
        # print(output_sequence)
    return output_sequence

def get_data_for_digit_test(source,digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    # print(all_samples)
    # print(source[0][16])
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        # print(batch_samples)
        output_sequence.append({
            # x，类型为tff.TensorType(tf.float32, [None, 784])
            # 一个0维长度不确定、1维长度为784的二维浮点数张量
            # 代表输入的样本维长度不确定是因为我们可以一批输入任意多个样本，通过矩阵运算来加速模型训练
            'x':
                np.array(
                    [source[0][i].flatten() / 255.0 for i in batch_samples],  # 灰度归一化
                    dtype=np.float32),
            # y，类型为tff.TensorType(tf.int32, [None])
            # 一个0维长度不确定的一维整数张量
            # 代表输出的标签，y的0维长度应当和x的相同，表示样本和标签一一对应
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
        # print(output_sequence)
    return output_sequence

federated_test_data = [get_data_for_digit_test(mnist_test, d) for d in range(10)]

BATCH_TYPE = tff.NamedTupleType([('x', tff.TensorType(tf.float32, [None, 784])),
                                 ('y', tff.TensorType(tf.int32, [None]))])

MODEL_TYPE = tff.NamedTupleType([('weights', tff.TensorType(tf.float32, [784, 10])),
                                 ('bias', tff.TensorType(tf.float32, [10]))])

# 这里声明了是tf computation
# 定义损失函数
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    #回归模型
    predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    #每一行求和 张量深度-1  然后所有书相加求平均
    return -tf.reduce_mean(
        tf.reduce_sum(tf.one_hot(batch.y, 10) * tf.math.log(predicted_y), axis=[1]))  # 求标签对应的相对概率值


#Define initial model
initial_model = {
    'weights': np.zeros([784, 10], dtype=np.float32),
    'bias': np.zeros([10], dtype=np.float32)
}

#define Gradient descent on a single batch
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):

    # tff.utils.create_variables声明变量类型
    # Define a group of model variables and set them to `initial_model`.
    model_vars = tff.utils.create_variables('v', MODEL_TYPE)
    # print(type(model_vars))
    # print(model_vars)
    init_model = tff.utils.assign(model_vars, initial_model)

    # Perform one step of gradient descent using loss from `batch_loss`.
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    # tf.gradients(batch_loss(model_vars, batch), x)
    with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))

    # Return the model vars after performing this gradient descent step.
    with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)

#Gradient descent on a sequence of local data
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)

@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):

    # Mapping function to apply to each batch.
    # batch_fn改成tf_computation来包装，是可以正常运行的
    @tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        learning_rate = 0.1
        return batch_train(model, batch, learning_rate)
    return tff.sequence_reduce(all_batches, initial_model, batch_fn)

# Local eval
@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    # tff.sequence_sum Replace with `tff.sequence_average()` once implemented.
    return tff.sequence_sum(
        tff.sequence_map(
            #？？？？？
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))

#Define fed trainning
SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER, all_equal=True)
SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER, all_equal=True)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)
CLIENT_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.CLIENTS)


@tff.federated_computation(
    SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    return tff.federated_mean(
        tff.federated_map(
            local_train,
            [tff.federated_broadcast(model),
            tff.federated_broadcast(learning_rate),
            data]))

@tff.federated_computation(
    SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_model(model, learning_rate, data):
    return tff.federated_map(
        local_train,
        [tff.federated_broadcast(model),
        tff.federated_broadcast(learning_rate),
        data])

@tff.federated_computation(CLIENT_MODEL_TYPE)
def federated_mean(model):
    return tff.federated_mean(model)

@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))

def cluster_train(roc, threshold,train_data,model,length,common_point,Interface):
    # 下面开始真正的fed训练
    # model = initial_model
    learning_rate = 0.1

    for round_num in range(roc):

        client_model = federated_model(model, learning_rate, train_data)
        # print(client_model[3])

        channel = rayleigh.Rayleigh_2(7850,threshold,1,length-common_point)
        for i in range(length-common_point):
            for j in range(784):
                for k in range(10):
                    num = (j - 1) * 10 + k - 1
                    client_model[i].weights[j][k] = channel[i][num] * client_model[i].weights[j][k]
            for l in range(10):
                client_model[i].bias[l] = channel[i][7840 + l] * client_model[i].bias[l]
        model = federated_mean(client_model)

        # 取平均
        map_sum = rayleigh.Rayleigh_3(channel,length-common_point)
        for j in range(784):
            for k in range(10):
                num = (j - 1) * 10 + k - 1
                temp1 = length / (map_sum[num] + common_point)
                model.weights[j][k] = model.weights[j][k] * temp1
        for l in range(10):
            temp2 = length / (map_sum[7840 + l] + common_point)
            model.bias[l
            ] = model.bias[l] * temp2
    # learning_rate = learning_rate*0.9
    return model

federated_train_data_sum = [get_data_for_digit(mnist_train, d) for d in range(10)]

Interface = [[9, 6, 7, 8, 5], [6, 4, 3], [1, 2, 4, 0]]  # P点放最后
common_point = [1, 1, 1]   #对应P数目  P点放最后
threshold_distribute = [rayleigh.threshold90,rayleigh.threshold90,rayleigh.threshold90]

federated_train_data = []

for i in range(len(Interface)):
    federated_train_data.append([get_data_for_digit(mnist_train, d) for d in Interface[i]])#
# print(federated_train_data[1][1])


# model_final = cluster_train(1, rayleigh.threshold100, federated_train_data[0], initial_model, len(Interface[0]),common_point[0])
model_final = initial_model
for i in range(100):
    for j in range(len(Interface)):
        model_final = cluster_train(1, threshold_distribute[j], federated_train_data[j], model_final, len(Interface[j]),
                                    common_point[j],Interface)
        loss = federated_eval(model_final, federated_train_data_sum)
    print('{}'.format(loss))



loss = federated_eval(model_final, federated_test_data)
print('{}'.format(loss))












