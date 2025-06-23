# 导入操作系统相关功能、系统功能和异常追踪模块
import os, sys, traceback
# 导入numpy库，用于数值计算和数组操作
import numpy as np
# 导入pandas库，用于数据处理和分析
import pandas as pd
# 从sklearn的预处理模块导入标签编码器
from sklearn.preprocessing import LabelEncoder

# 导入日志模块
import logging
# 配置日志格式：时间戳、毫秒、日志级别和消息内容
logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# 定义Criteo数据集预处理函数
def preprocess_criteo():
    # 数据源说明：Criteo广告展示挑战赛数据集
    # source: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
    # 设置数据文件路径
    data_path = "./data/criteo/train.txt"
    # 断言检查数据文件是否存在，如果不存在则输出错误信息
    assert os.path.exists(data_path), \
        "Please download dac.tar.gz and extract to ./criteo/ in advance"

    # 加载数据
    logging.info("Reading data...")
    # 原始代码（已注释掉）：读取全部数据
    # df = pd.read_csv(data_path, sep='\t', header=None)
    # 修改后的代码：只读取部分数据以加快处理速度
    df = pd.read_csv(data_path, sep='\t', header=None, nrows=1000000)  # 限制读取100万行
    # 设置列名：第一列为标签，接下来13列为数值特征(I1-I13)，后26列为类别特征(C14-C39)
    df.columns = ["labels"] + ["I%d"%i for i in range(1,14)] + ["C%d"%i for i in range(14,40)]

    # 预处理稠密和稀疏数据
    logging.info("Preprocessing data...")
    # 提取标签列
    labels = df["labels"]
    # 提取所有以'I'开头的列作为稠密特征（数值型特征）
    dense_feats =  [col for col in df.columns if col.startswith('I')]
    # 提取所有以'C'开头的列作为稀疏特征（类别型特征）
    sparse_feats = [col for col in df.columns if col.startswith('C')]
    # 处理稠密特征
    dense_feats = process_dense_feats(df, dense_feats)
    # 处理稀疏特征
    sparse_feats = process_sparse_feats(df, sparse_feats)

    # 将数据集分割为训练集和验证集
    # 获取数据总量
    num_data = dense_feats.shape[0]
    # 验证集大小为总数据量的十分之一
    num_valid = num_data // 10
    # 生成随机排列索引，用于随机划分数据集
    perm = np.random.permutation(num_data)
    # 提取训练集的稠密特征，并转换为float32类型
    train_dense = dense_feats.iloc[perm[:-num_valid]].astype(np.float32)
    # 提取验证集的稠密特征，并转换为float32类型
    valid_dense = dense_feats.iloc[perm[-num_valid:]].astype(np.float32)
    # 提取训练集的稀疏特征，并转换为int32类型
    train_sparse = sparse_feats.iloc[perm[:-num_valid]].astype(np.int32)
    # 提取验证集的稀疏特征，并转换为int32类型
    valid_sparse = sparse_feats.iloc[perm[-num_valid:]].astype(np.int32)
    # 提取训练集的标签，并转换为float32类型
    train_labels = labels.iloc[perm[:-num_valid]].astype(np.float32)
    # 提取验证集的标签，并转换为float32类型
    valid_labels = labels.iloc[perm[-num_valid:]].astype(np.float32)

    # 保存处理后的数据
    logging.info("Saving data...")
    # 记录训练集数据大小信息
    logging.info(
        f"Train size: guest[{train_dense.shape}] " + 
        f"host[{train_sparse.shape}] " + 
        f"labels[{train_labels.shape}]")
    # 将训练集数据保存为npz格式
    # guest表示稠密特征（数值型），host表示稀疏特征（类别型）
    np.savez("./criteo/train.npz",
             guest=train_dense,
             host=train_sparse,
             labels=train_labels)
    # 记录验证集数据大小信息
    logging.info(
        f"Valid size: guest[{valid_dense.shape}] " + 
        f"host[{valid_sparse.shape}] " + 
        f"labels[{valid_labels.shape}]")
    # 将验证集数据保存为npz格式
    np.savez("./criteo/valid.npz",
             guest=valid_dense,
             host=valid_sparse,
             labels=valid_labels)

    # 记录预处理完成信息
    logging.info("Data preprocessing done")


# 定义处理稠密特征（数值型特征）的函数
def process_dense_feats(data, feats):
    # 记录正在处理的特征列表
    logging.info(f"Processing feats: {feats}")
    # 创建数据副本以避免修改原始数据
    d = data.copy()
    # 只保留需要的特征列，并将缺失值填充为0.0
    d = d[feats].fillna(0.0)
    # 对每个特征进行对数变换处理
    for f in feats:
        # 对大于-1的值应用对数变换：log(x+1)，否则保持为-1
        # 对数转换可以压缩数据范围，使分布更接近正态分布
        d[f] = d[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)    #对数转换可以压缩数据范围
    # 返回处理后的数据
    return d


# 定义处理稀疏特征（类别型特征）的函数
def process_sparse_feats(data, feats):
    """预处理稀疏(类别型)特征
    
    Args:
        data: 原始数据DataFrame
        feats: 需要处理的稀疏特征列表
        
    Returns:
        处理后的DataFrame，其中稀疏特征已被编码为连续整数
    """
    # 记录正在处理的特征列表
    logging.info(f"Processing feats: {feats}")
    
    # 创建数据副本以避免修改原始数据
    d = data.copy()
    
    # 只保留需要的特征列，并将缺失值填充为"-1"
    d = d[feats].fillna("-1")
    
    # 对每个特征列进行标签编码
    for f in feats:
        # 创建标签编码器
        label_encoder = LabelEncoder()  #这里直接把类别转为数字，而不是one-hot编码，可能是因为类别太多
        # 将类别值转换为整数编码
        d[f] = label_encoder.fit_transform(d[f])
    
    # 特征偏移处理，确保不同特征的编码值不重叠
    feature_cnt = 0
    for f in feats:
        # 将当前特征的编码值整体偏移feature_cnt
        d[f] += feature_cnt
        # 更新偏移量，加上当前特征的唯一值数量
        feature_cnt += d[f].nunique()
    
    # 返回处理后的数据
    return d

# 主程序入口
if __name__ == "__main__":
        # 调用Criteo数据预处理函数
        preprocess_criteo()
