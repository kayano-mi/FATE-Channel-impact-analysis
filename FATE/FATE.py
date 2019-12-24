#   NON I.I.D


from __future__ import absolute_import, division, print_function
import rayleigh
import os


# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
# # os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error
#显示等级

# # 选择cpu工作
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置定量的GPU使用量:
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
# session = tf.Session(config=config)

# 设置最小的GPU使用量:
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



import collections
import numpy as np
import minpy.numpy as mnp
np.set_printoptions(threshold = 1e2)#设置打印数量的阈值
# import numpy.random as npr
from six.moves import range
import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

NUM_EXAMPLES_PER_USER = 5000  # max = 5421
BATCH_SIZE = 1000  #小于num且最好为num的整数倍

# train-images-idx3-ubyte.gz
# 训练集图片 - 55000 张 训练图片, 5000 张 验证图片
# train-labels-idx1-ubyte.gz
# 训练集图片对应的数字标签
# t10k-images-idx3-ubyte.gz
# 测试集图片 - 10000 张 图片
# t10k-labels-idx1-ubyte.gz
# 测试集图片对应的数字标签

# 入口数据处理

def get_data_for_digit(source, digit):
    output_sequence = []

    # print([i for i in range(1, 11)])
    # print([i * 2 for i in range(1, 11)])
    # print([i * i for i in range(1, 11)])
    # print([str(i) for i in range(1, 11)])
    # print([i for i in range(1, 11) if i % 2 == 0])
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]

    # print(all_samples)
    # print(source[0][16])
    for i in range(0, NUM_EXAMPLES_PER_USER,BATCH_SIZE):
        # print(min(len(all_samples), NUM_EXAMPLES_PER_USER))
        batch_samples = all_samples[i:i + BATCH_SIZE]
        # print(batch_samples)
        output_sequence.append({
            # x，类型为tff.TensorType(tf.float32, [None, 784])
            # 一个0维长度不确定、1维长度为784的二维浮点数张量
            # 代表输入的样本维长度不确定是因为我们可以一批输入任意多个样本，通过矩阵运算来加速模型训练
            'x':
            np.array(
                [source[0][i].flatten() / 255.0 for i in batch_samples], #灰度归一化
                dtype=np.float32),
            #y，类型为tff.TensorType(tf.int32, [None])
            # 一个0维长度不确定的一维整数张量
            # 代表输出的标签，y的0维长度应当和x的相同，表示样本和标签一一对应
            'y':
            np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
        # print(output_sequence)
    return output_sequence

federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]
federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]


# 定义端无关变量 x,y,weights,bias
# tff.TensorType张量类型
# tff.SequenceType列表类型
# tff.NamedTupleType就是元素可以带有key的tuple
# tff.NamedTupleType接受三种类型的输入：list，tuple和collections.OrderedDict（Python标准库）
# 也就是collection.namedtuple生成的subclass生成的对象，一种有序的字典dict类型
# tff.FunctionType函数类型
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

# sample_batch = federated_train_data[6]
# print(sample_batch)
# print("init batch loss:", batch_loss(initial_model, sample_batch))


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

    with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))
        # print(optimizer.compute_gradients(batch_loss(model_vars, batch)))

    # Return the model vars after performing this gradient descent step.
    with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)


# #Single client train
# model = initial_model
# losses = []
# for _ in range(5):
#     model = batch_train(model, sample_batch, 0.1)
#     losses.append(batch_loss(model, sample_batch))
# print("5 loops loss:", losses)

#Gradient descent on a sequence of local data
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)


# 端有关函数
# 如果函数是tff.federated_开头的，就是端有关的，会规定输入输出的存放位置。
# 其他的函数即是端无关的，输入输出的类型也是端无关类型（有些也可以用在端有关场景）
# 注意：
# 1、函数调用的语法就和调用普通的python函数一样，但是要注意输入参数的类型是否匹配（或可以自动转换）。
# 2、tff.tf_computation包装的函数不能调用tff.federated_computation包装的，但反过来可以（无关不能调用有关）。
# 3、tff.federated_computation包装的函数输入参数不一定都是Federated Type，但其中的表达的逻辑是端有关的。
@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):

    # Mapping function to apply to each batch.
    # batch_fn改成tf_computation来包装，是可以正常运行的
    @tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        learning_rate = 0.1
        return batch_train(model, batch, learning_rate)
    return tff.sequence_reduce(all_batches, initial_model, batch_fn)


# locally_trained_model_0= local_train(initial_model, 0.1, federated_train_data[1])
# print("local_train", local_train.type_signature)



# Local eval
@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    # tff.sequence_sum Replace with `tff.sequence_average()` once implemented.
    return tff.sequence_sum(
        tff.sequence_map(
            #？？？？？
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))

# print("local_eval", local_eval.type_signature)
# print('initial_model loss [num 5] =', local_eval(initial_model, federated_train_data[5]))
# print('locally_trained_model loss [num 5] =', local_eval(locally_trained_model_0, federated_train_data[5]))
# print('initial_model loss [num 0] =', local_eval(initial_model, federated_train_data[0]))
# print('locally_trained_model loss [num 0] =', local_eval(locally_trained_model_0, federated_train_data[0]))
# print(federated_train_data[5])

# 端有关类型
# 1、显式地定义数据值应该存放在C端还是S端（Placement）
#  tff.SERVER和tff.CLIENTS，也就是定义了S端和C端，当常数用。
# 2、定义这个数据是否全局一致（All equal?）
#  placement（必填），必须是tff.SERVER和tff.CLIENTS这些Placement type
#  all_equal（可选，默认为None）。类型为bool，代表着这份数据是否全局统一
#  还是可以有不同的值。如果没有指定all_equal，它会根据placement的值来选择。
#  默认情况下placement=tff.SERVER时all_equal=True，反之为False。

# Federated types的类型表示格式为T@G或{T}@G，其中T为TFF的数据类型，G为存放的位置，
# 花括号{}表示非全局唯一，而没有花括号就表示全局唯一，即all_equal=True

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

# @tff.federated_computation(CLIENT_MODEL_TYPE)
# def federated_rayleigh(model):
#     # Rayleigh1 = rayleigh.channel1
#     # print(Rayleigh1)
#     # weight1 = model[1].weights.flatten()
#     # bias1 = model[1]
#     # print(model[1])
#     # model = np.multiply(Rayleigh,model)
#     return model

# print("federated_train", federated_train.type_signature)

# Define server model
@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))


# 下面开始真正的fed训练
model = initial_model
learning_rate = 0.1

for round_num in range(200):

    '''
    # 每一轮，把大家的模型分别更新一下，取平均之后拿回来
    # print(model)
    # model = federated_train(model, learning_rate, federated_train_data)
    # print(model) #更新后P端模型
    '''

    loss = federated_eval(model, federated_train_data)
    print('{}'.format(loss))

    client_model = federated_model(model, learning_rate, federated_train_data)
    # print(client_model[1])
    # print(client_model) #10个C端模型输出

    channel = rayleigh.Rayleigh_2(7850,rayleigh.threshold70,1,10)

    #在此处修改C端上传模型的值
    #大循环10个C端用户
    for i in range(10):

        # weights赋值
        for j in range(784):
            for k in range(10):
                num = (j-1)*10+k-1
                client_model[i].weights[j][k] = channel[i][num]*client_model[i].weights[j][k]

        # bias赋值
        for l in range(10):
            client_model[i].bias[l] = channel[i][7840+l]*client_model[i].bias[l]

    # print(client_model[9])
    # tempb = np.multiply(rayleigh.channel1, client_model[1].bias[1])
    # client_model[1].bias =np.multiply(rayleigh.channel1, client_model[1].bias[1])
    # print(client_model[1].bias)
    # print(tempb)
    # client_model_before = federated_rayleigh(client_model)

    model = federated_mean(client_model)
    # print(model.bias)


    # model = model/map_sum
    map_sum = rayleigh.Rayleigh_3(channel,10)
    for j in range(784):
        for k in range(10):
            num = (j - 1) * 10 + k - 1
            # print(map_sum[num])

            if map_sum[num] == 0:
                map_sum[num] = 0.0000001


            temp1 = 10.0 / map_sum[num]
            # print(temp1)
            model.weights[j][k] = model.weights[j][k] * temp1

        # bias赋值
    for l in range(10):
        temp2 = 10.0 / map_sum[7840 + l]
        model.bias[l] = model.bias[l] * temp2
    # print(model.bias)

    # # 更新一下学习率
    # learning_rate = learning_rate
    # # 算个loss输出一下

loss = federated_eval(model, federated_test_data)
print('{}'.format(loss))











# Functions
# federated_aggregate(...): Aggregates value from tff.CLIENTS to tff.SERVER.
# federated_apply(...): Applies a given function to a federated value on the tff.SERVER.
# federated_broadcast(...): Broadcasts a federated value from the tff.SERVER to the tff.CLIENTS.
# federated_collect(...): Returns a federated value from tff.CLIENTS as a tff.SERVER sequence.
# federated_computation(...): Decorates/wraps Python functions as TFF federated/composite computations.
# federated_map(...): Maps a federated value on tff.CLIENTS pointwise using a mapping function.
# federated_mean(...): Computes a tff.SERVER mean of value placed on tff.CLIENTS.
# federated_reduce(...): Reduces value from tff.CLIENTS to tff.SERVER using a reduction op.
# federated_sum(...): Computes a sum at tff.SERVER of a value placed on the tff.CLIENTS.
# federated_value(...): Returns a federated value at placement, with value as the constituent.
# federated_zip(...): Converts an N-tuple of federated values into a federated N-tuple value.
# sequence_map(...): Maps a TFF sequence value pointwise using a given function mapping_fn.
# sequence_reduce(...): Reduces a TFF sequence value given a zero and reduction operator op.
# sequence_sum(...): Computes a sum of elements in a sequence.
# tf_computation(...): Decorates/wraps Python functions and defuns as TFF TensorFlow computations.
# to_type(...): Converts the argument into an instance of tff.Type.
# to_value(...): Converts the argument into an instance of the abstract class tff.Value.

