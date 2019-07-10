#!/usr/bin/python3
from __future__ import division
from __future__ import print_function

from config import FLAGS
from train import train_loop, test
from utils_siamese import get_model_info_as_str, \
    check_flags, extract_config_code, convert_long_time_to_str
from data_siamese import SiameseModelData
from dist_sim_calculator import DistSimCalculator
from models_factory import create_model
from saver import Saver
import tensorflow as tf
from time import time
import os


def main():
    t = time()
    conf_code = extract_config_code()
    check_flags()
    print(get_model_info_as_str())
    data = SiameseModelData(FLAGS.dataset_train)
    dist_sim_calculator = DistSimCalculator(
        FLAGS.dataset_train, FLAGS.ds_metric, FLAGS.ds_algo)
    model = create_model(FLAGS.model, data.input_dim(),
                         data, dist_sim_calculator)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = Saver(sess)
    sess.run(tf.global_variables_initializer())
    train_costs, train_times = \
        train_loop(data, model, saver, sess)
    test(data, model, saver, sess)
    saver.save_conf_code(conf_code)
    overall_time = convert_long_time_to_str(time() - t)
    print(overall_time, saver.get_log_dir())
    saver.save_overall_time(overall_time)
    return train_costs, train_times


if __name__ == '__main__':
    main()
