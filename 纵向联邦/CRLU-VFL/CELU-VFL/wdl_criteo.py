#-*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np

from python.fdl.federation import Federation
from python.fdl.tensorflow.vfl_graph import FLGuestGraph, FLHostGraph

import os, sys, traceback
import logging

def wdl_criteo_fed(args):
    # 使用Adagrad优化器，学习率从args参数获取
    optimizer = tf.train.AdagradOptimizer(args.eta)
    
    # 判断当前参与方是guest还是host
    if args.party == "guest":
        # 定义guest方的输入占位符(13维连续特征)
        guest_input = tf.placeholder(tf.float32, [None, 13], name="guest-input")
        # 定义二分类标签占位符
        labels = tf.placeholder(tf.float32, [None, 1], name="labels")
        # 创建FLGuestGraph对象，依赖guest_input
        graph = FLGuestGraph(deps=[guest_input])
        
        # guest方的底层模型(处理连续特征)
        guest_act = wdl_criteo_guest_bottom(guest_input, device=args.device)
        
        # 获取host方的256维激活值(通过联邦通信)
        host_act = graph.remote_bottom("HostAct", shape=[None, 256])
        # 构建顶层模型(计算损失和指标)
        top_model = wdl_criteo_top_model(labels, guest_act, host_act, device=args.device)
        
        # 联邦训练操作(返回训练op和梯度信息)
        train_op, _, devs_and_acts = graph.minimize(
            optimizer, top_model["loss"], 
            return_grads=True)
        # 预测操作(输出logits)
        pred_op = graph.predict(top_model["logits"])
        
        # 返回guest方需要的所有计算节点
        return guest_input, labels, top_model, train_op, pred_op
    else:
        # host方的输入占位符(26维离散特征)
        host_input = tf.placeholder(tf.int32, [None, 26], name="host-input")
        # 创建FLHostGraph对象
        graph = FLHostGraph()
        # host方的底层模型(处理离散特征)
        host_act = wdl_criteo_host_bottom(host_input, device=args.device)
        # 向guest方发送激活值
        graph.send_bottom("HostAct", host_act)
        # 联邦训练操作
        train_op, _, devs_and_acts = graph.minimize(optimizer, return_grads=True)
        # 预测操作
        pred_op = graph.predict()
        
        # 返回host方需要的所有计算节点
        return host_input, train_op, pred_op


def wdl_criteo_fed_with_local_update(args):
    """带有本地更新的WDL模型联邦训练函数
    Args:
        args: 包含训练参数的对象(如学习率eta、设备device等)
    Returns:
        根据参与方类型返回不同的计算节点和操作
    """
    optimizer = tf.train.AdagradOptimizer(args.eta)
    if args.party == "guest":
        # Guest方(标签持有方)的处理逻辑
        guest_input = tf.placeholder(tf.float32, [None, 13], name="guest-input")  # 连续特征输入
        labels = tf.placeholder(tf.float32, [None, 1], name="labels")  # 标签
        graph = FLGuestGraph(deps=[guest_input])  # 创建Guest图
        
        # Guest方的底层模型(处理连续特征)
        guest_act = wdl_criteo_guest_bottom(guest_input, device=args.device)
        
        # 获取Host方的激活值(通过联邦通信)
        host_act = graph.remote_bottom("HostAct", shape=[None, 256])
        # 构建顶层模型(计算损失和指标)
        top_model = wdl_criteo_top_model(labels, guest_act, host_act, device=args.device)
        
        # 联邦训练操作
        train_op, _, devs_and_acts = graph.minimize(
            optimizer, top_model["loss"], 
            return_grads=True)
        pred_op = graph.predict(top_model["logits"])  # 预测操作
        
        # 本地更新相关操作
        host_dev = list(filter(lambda x: x[1] is host_act, devs_and_acts))[0][0]  # 获取Host方梯度
        cache_host_act = tf.placeholder(tf.float32, host_act.shape, name="CacheAct")  # 缓存Host激活值
        cache_host_dev = tf.placeholder(tf.float32, host_dev.shape, name="CacheDev")  # 缓存Host梯度
        
        # 本地更新模型(使用缓存的Host信息)
        local_top_model = wdl_criteo_top_model(
            labels, guest_act, cache_host_act, 
            device=args.device)
        local_train_op, ins_weights = graph.local_minimize(
            optimizer, local_top_model["ins_loss"], 
            {"HostAct" : cache_host_act}, 
            {"HostAct" : cache_host_dev}, 
            args.sim_thres)
        
        return guest_input, labels, top_model, train_op, pred_op, \
            host_act, host_dev, cache_host_act, cache_host_dev, \
            local_top_model, local_train_op
    else:
        # Host方(特征持有方)的处理逻辑
        host_input = tf.placeholder(tf.int32, [None, 26], name="host-input")  # 离散特征输入
        graph = FLHostGraph()  # 创建Host图
        host_act = wdl_criteo_host_bottom(host_input, device=args.device)  # Host底层模型
        graph.send_bottom("HostAct", host_act)  # 发送激活值给Guest
        
        # 联邦训练操作
        train_op, _, devs_and_acts = graph.minimize(optimizer, return_grads=True)
        pred_op = graph.predict()
        
        # 本地更新相关操作
        host_dev = list(filter(lambda x: x[1] is host_act, devs_and_acts))[0][0]  # 获取本地梯度
        cache_host_act = tf.placeholder(tf.float32, host_act.shape, name="CacheAct")  # 缓存激活值
        cache_host_dev = tf.placeholder(tf.float32, host_dev.shape, name="CacheDev")  # 缓存梯度
        
        # 本地更新操作
        local_train_op, ins_weights = graph.local_minimize(
            optimizer, 
            {"HostAct" : cache_host_dev}, 
            {"HostAct" : cache_host_act}, 
            args.sim_thres)
        
        return host_input, train_op, pred_op, \
            host_act, host_dev, cache_host_act, cache_host_dev, \
            local_train_op


def wdl_criteo_guest_bottom(guest_input, device="/gpu:0"):
    """Guest方底层模型(处理连续特征)
    Args:
        guest_input: 输入特征张量
        device: 计算设备
    Returns:
        经过三层全连接网络后的激活值
    """
    with tf.variable_scope("guest_bottom", dtype=tf.float32, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            dim = guest_input.shape[1]
            W1 = tf.get_variable(name='W1', shape=[dim, 256])  # 第一层权重
            W2 = tf.get_variable(name='W2', shape=[256, 256])  # 第二层权重
            W3 = tf.get_variable(name='W3', shape=[256, 256])  # 第三层权重

            act1 = tf.nn.relu(tf.matmul(guest_input, W1))  # 第一层激活(ReLU)
            act2 = tf.nn.relu(tf.matmul(act1, W2))  # 第二层激活(ReLU)
            act3 = tf.matmul(act2, W3)  # 第三层激活(无激活函数)

            return act3


def wdl_criteo_host_bottom(host_input, device="/gpu:0", num_buckets=1000000, embedding_size=128):
    """Host方底层模型(处理离散特征)
    Args:
        host_input: 输入特征张量
        device: 计算设备
        num_buckets: 哈希桶数量
        embedding_size: 嵌入维度
    Returns:
        经过嵌入层和两层全连接网络后的激活值
    """
    with tf.variable_scope("host_bottom", dtype=tf.float32, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            # 嵌入层
            embedding = tf.get_variable(name="Embedding", shape=(num_buckets, embedding_size))
            dim = host_input.shape[1] * embedding_size
            W2 = tf.get_variable(name='W1', shape=[dim, 256])  # 第一层权重
            W3 = tf.get_variable(name='W2', shape=[256, 256])  # 第二层权重
            
            # 特征哈希和嵌入查找
            hash_buckets = tf.strings.to_hash_bucket_fast(
                tf.strings.as_string(host_input), num_buckets)
            act1 = tf.nn.embedding_lookup(embedding, hash_buckets)
            act1 = tf.reshape(act1, (-1, dim))  # 展平嵌入结果
            
            act2 = tf.nn.relu(tf.matmul(act1, W2))  # 第一层激活(ReLU)
            act3 = tf.matmul(act2, W3)  # 第二层激活(无激活函数)

            return act3


def wdl_criteo_top_model(labels, guest_output, host_output, device="/gpu:0"):
    """顶层模型(计算预测和损失)
    Args:
        labels: 真实标签
        guest_output: Guest方模型输出
        host_output: Host方模型输出
        device: 计算设备
    Returns:
        包含预测结果、概率、损失和准确率的字典
    """
    with tf.variable_scope("top", dtype=tf.float32, reuse=tf.AUTO_REUSE, 
                           initializer=tf.random_normal_initializer(stddev=0.01)):
        with tf.device(device):
            concat = tf.concat((guest_output, host_output), 1)  # 连接双方输出
            W1 = tf.get_variable(name='W1', shape=[256 * 2, 1])  # 顶层权重
            
            logits = tf.matmul(concat, W1)  # 预测值
            proba = tf.nn.sigmoid(logits)  # 概率值
            ins_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels)  # 实例级损失
            loss = tf.reduce_mean(ins_loss)  # 平均损失
            corrects = tf.math.equal(tf.cast(logits >= 0, tf.float32), labels)  # 正确预测
            acc = tf.reduce_mean(tf.cast(corrects, tf.float32))  # 准确率

            return {
                "logits": logits,  # 原始预测值
                "proba": proba,  # 概率值(0-1)
                "ins_loss": ins_loss,  # 实例级损失
                "loss": loss,  # 平均损失
                "acc": acc,  # 准确率
            }

