# coding:UTF-8
# @Time: 2022/10/10 16:46
# @Author: Lulu Cao
# @File: configs.py
# @Software: PyCharm
'''
parameters config
'''


# 选取的训练和测试数据集的数量
N=1200
ntrain = 20 #1000
ntest = 20 #200


# bath_size
batch_size = 20
bsize = 4

# 基本的训练参数
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)



# sub用于下采样分辨率，比如数据集生成的是256*256，将sub设置为4即可得到64*64，在验证分辨率不变性的实验里用到
sub = 1
# 分辨率设置，该数据集是256*256的
S = 256
# Fourier层中的两个参数
modes = 12
width = 20

# 用于设置FNO输入的时间序列长度，本实验的输入是T在[0,10)时的w
record_steps=200
T=50 #实验数据是[0,50)
T_in = 10 #输入是[0,10)
T_in = int(T_in/T*record_steps)

T_after_delta = 40 #输出是[10,10+T_after_delta)
T_after_delta = int(T_after_delta/T*record_steps)
step = 1


