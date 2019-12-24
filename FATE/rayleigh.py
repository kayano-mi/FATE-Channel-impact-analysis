import numpy as np

import numpy.random as npr
import tensorflow as tf
import tensorflow_federated as tff
np.set_printoptions(threshold = 1e6)#设置打印数量的阈值
# length = 7850

#保留值的概率为：
#p(am)=1-exp(-0.5*threshold^2)
#threshold = squa(-2*ln(1-p))




#p = 1
threshold100 = 10
# p = 0.9
threshold90 = 2.145966026289347#23963618357029‬
# p = 0.8
threshold80 = 1.794122577994101#4802839975948169‬
# p = 0.7
threshold70 = 1.551755653655520#5943289659103096‬
# p = 0.6
threshold60 = 1.353728726055671#070381037018586
# p = 0.5
threshold50 = 1.177410022515474#6910115693264597
# p = 0.45
threshold45 = 1.093468793112652#5289569326575534‬
# p = 0.4
threshold40 = 1.0107676525947896#431381113653917‬
# p = 0.35
threshold35 = 0.92820570574895116#551914773074033‬
# p = 0.3
threshold30 = 0.8446004309005914#5305799362259457‬
# p = 0.2
threshold20 = 0.6680472308365775#3571759305652113‬



def Rayleigh(scale,size):

    Amplitude = npr.rayleigh(scale,size)

    return Amplitude

def Rayleigh_2(length,threshold,scale,num_channel):

    channel = []
    for i in range(num_channel):
        channel.append(Rayleigh(scale, length))
    for i in range(num_channel):
        channel[i][channel[i] < threshold] = 1
        channel[i][channel[i] != 1] = 0

    return channel

#已知信道状态保留的信道数
def Rayleigh_3(channel,sum_num):
    map_sum = channel[sum_num-1]
    for i in range(sum_num-1):
        map_sum = map_sum + channel[i]
    return map_sum

# channel = Rayleigh_2(10,2,1,5)
# map_sum = Rayleigh_3(channel,5)
# print(channel)
# print(map_sum)

# centList=1
# while (len(centList) < 2):
#




# import tensorflow as tf
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# for i in range(0, 100, 10):
#     print(i)



    # channel1 = channel1.transpose()
    #
    # channel2 = Rayleigh(scale, length)
    # channel2[channel2 < threshold] = 1
    # channel2[channel2 != 1] = 0
    #
    # channel3 = Rayleigh(scale, length)
    # channel3[channel3 < threshold] = 1
    # channel3[channel3 != 1] = 0
    #
    # channel4 = Rayleigh(scale, length)
    # channel4[channel4 < threshold] = 1
    # channel4[channel4 != 1] = 0
    #
    # channel5 = Rayleigh(scale, length)
    # channel5[channel5 < threshold] = 1
    # channel5[channel5 != 1] = 0
    #
    # channel6 = Rayleigh(scale, length)
    # channel6[channel6 < threshold] = 1
    # channel6[channel6 != 1] = 0
    #
    # channel7 = Rayleigh(scale, length)
    # channel7[channel7 < threshold] = 1
    # channel7[channel7 != 1] = 0
    #
    # channel8 = Rayleigh(scale, length)
    # channel8[channel8 < threshold] = 1
    # channel8[channel8 != 1] = 0
    #
    # channel9 = Rayleigh(scale, length)
    # channel9[channel9 < threshold] = 1
    # channel9[channel9 != 1] = 0
    #
    # channel10 = Rayleigh(scale, length)
    # channel10[channel10 < threshold] = 1
    # channel10[channel10 != 1] = 0

    # channel = [channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8, channel9, channel10]