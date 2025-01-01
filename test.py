from gpr_net import gpr_net
# from train import data_loder_npy
import tensorflow as tf
import numpy as np
import os
import random
from loss_fun import metrics
from loss_fun import loss as gpr_loss

def data_loder_npy():
    x_test = np.load('/home/chb/gpr/Attention/npy/test_data_ys.npy')
    y_test = np.load('/home/chb/gpr/Attention/npy/test_label_ys.npy')
    return x_test, y_test


def main():
    # 设置随机性
    seed = 1921
    np.random.seed(seed)  # seed是一个固定的整数即可
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)  # tensorflow2.0版本的设置，较早版本的设置方式不同，可以自查

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    x_test = np.load('/home/chb/gpr/Attention/npy/test_data_ys.npy')
    y_test = np.load('/home/chb/gpr/Attention/npy/test_label_ys.npy')
    # x_train, y_train, x_val, y_val, x_test, y_test = data_loder_npy(5569, 6264)

    # 载入模型
    model_path = r'/home/chb/gpr/Attention/model_ys_duotou/data100_500_epoch500_mse_ys_duotou.h5'
    model = gpr_net()
    model.load_weights(model_path)
    model.compile(loss=gpr_loss(k=0.5), optimizer='adam', metrics=['mse', metrics()])

    # 测试
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=1)
    print(loss_and_metrics)


if __name__ == '__main__':
    main()
