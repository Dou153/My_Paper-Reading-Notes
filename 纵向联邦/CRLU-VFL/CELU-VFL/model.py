# 导入numpy库，用于数值计算
import numpy as np
# 导入pandas库，用于数据处理和分析
import pandas as pd
# 导入matplotlib的pyplot模块，用于绘图
import matplotlib.pyplot as plt
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入datetime库，用于处理日期和时间
import datetime

# 导入PyTorch库
import torch
# 导入PyTorch的数据加载和处理模块
from torch.utils.data import DataLoader, Dataset, TensorDataset
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的函数式API
import torch.nn.functional as F
# 导入PyTorch的优化器
import torch.optim as optim

# 导入torchkeras库中的summary和Model函数，用于模型可视化和训练
from torchkeras import summary
from torchkeras import KerasModel as Model

# 导入sklearn的ROC AUC评分函数
from sklearn.metrics import roc_auc_score

# 导入warnings库
import warnings
# 忽略所有警告信息
warnings.filterwarnings('ignore')


# ## 数据准备

# 设置数据文件路径
file_path = './data/criteo'


# 定义数据准备函数
def prepared_data(file_path):
    
    # 读入训练集和验证集
    train_data = np.load(file_path + '/train.npz')
    valid_data = np.load(file_path + '/valid.npz')
    
    # 从NPZ文件中提取数据
    # guest是稠密特征（数值型），host是稀疏特征（类别型）
    trn_dense = train_data['guest']  # 训练集稠密特征
    trn_sparse = train_data['host']  # 训练集稀疏特征
    trn_y = train_data['labels']     # 训练集标签
    
    val_dense = valid_data['guest']  # 验证集稠密特征
    val_sparse = valid_data['host']  # 验证集稀疏特征
    val_y = valid_data['labels']     # 验证集标签
    
    # 构建特征列信息
    # 稠密特征列（数值型特征）
    dense_features = [{'name': f'I{i}', 'type': 'dense'} for i in range(trn_dense.shape[1])]
    # 稀疏特征列（类别型特征）
    sparse_features = [{'name': f'C{i}', 'type': 'sparse', 'feat_num': int(trn_sparse[:, i].max()) + 1, 'embed_dim': 8} 
                      for i in range(trn_sparse.shape[1])]
    
    # 特征列信息
    fea_col = [dense_features, sparse_features]
    
    # 返回特征列信息和数据集
    return fea_col, (trn_dense, trn_sparse, trn_y), (val_dense, val_sparse, val_y)


# ## 构建模型 - 纵向联邦学习架构

# 定义Guest方底层模型（处理连续特征）
class GuestBottom(nn.Module):
    """
    Guest方底层模型(处理连续特征)
    """
    def __init__(self, input_dim):
        super(GuestBottom, self).__init__()
        # 三层全连接网络
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        
        # 初始化权重
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.normal_(self.fc3.weight, std=0.01)
    
    def forward(self, x):
        # 第一层激活(ReLU)
        x = F.relu(self.fc1(x))
        # 第二层激活(ReLU)
        x = F.relu(self.fc2(x))
        # 第三层（无激活函数）
        x = self.fc3(x)
        return x


# 定义Host方底层模型（处理离散特征）
class HostBottom(nn.Module):
    """
    Host方底层模型(处理离散特征)
    """
    def __init__(self, sparse_feature_cols, num_buckets=1000000, embedding_size=128):
        super(HostBottom, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(num_buckets, embedding_size)
        # 初始化嵌入层权重
        nn.init.normal_(self.embedding.weight, std=0.01)
        
        # 计算输入维度
        self.input_dim = len(sparse_feature_cols) * embedding_size
        
        # 全连接层
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # 初始化权重
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        
        # 保存特征列信息和哈希桶数量
        self.sparse_feature_cols = sparse_feature_cols
        self.num_buckets = num_buckets
    
    def forward(self, x):
        # 将输入转换为长整型
        x = x.long()
        
        # 特征哈希和嵌入查找
        batch_size = x.shape[0]
        embeds = []
        
        for i in range(x.shape[1]):
            # 将特征值转换为字符串并哈希（模拟TF的哈希操作）
            # 在PyTorch中模拟tf.strings.to_hash_bucket_fast
            feature_values = x[:, i]
            # 使用PyTorch内置的哈希函数计算哈希值
            # 对每个值取绝对值后对num_buckets取模，确保在有效范围内
            hash_buckets = torch.remainder(feature_values * 2654435761, self.num_buckets)
            hash_buckets = hash_buckets.long()
            
            # 使用哈希值查找嵌入（模拟tf.nn.embedding_lookup）
            feature_embed = self.embedding(hash_buckets)
            embeds.append(feature_embed)
        
        # 拼接所有嵌入向量
        embeds = torch.cat(embeds, dim=1)
        # 重塑为二维张量
        embeds = embeds.view(batch_size, -1)
        
        # 第一层激活(ReLU)
        x = F.relu(self.fc1(embeds))
        # 第二层（无激活函数）
        x = self.fc2(x)
        
        return x


# 定义顶层模型
class TopModel(nn.Module):
    """
    顶层模型(计算预测和损失)
    """
    def __init__(self):
        super(TopModel, self).__init__()
        # 顶层权重
        self.fc = nn.Linear(256 * 2, 1)
        # 初始化权重
        nn.init.normal_(self.fc.weight, std=0.01)
    
    def forward(self, guest_output, host_output):
        # 连接双方输出
        concat = torch.cat((guest_output, host_output), dim=1)
        # 预测值
        logits = self.fc(concat)
        # 概率值
        proba = torch.sigmoid(logits)
        
        return {
            "logits": logits,  # 原始预测值
            "proba": proba,  # 概率值(0-1)
        }


# 定义完整的纵向联邦学习模型
class VFLModel(nn.Module):
    """
    纵向联邦学习模型
    """
    def __init__(self, feature_columns, num_buckets=1000000, embedding_size=128):
        super(VFLModel, self).__init__()
        # 分离稠密特征列和稀疏特征列
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        
        # Guest方底层模型
        self.guest_bottom = GuestBottom(len(self.dense_feature_cols))
        # Host方底层模型
        self.host_bottom = HostBottom(self.sparse_feature_cols, num_buckets, embedding_size)
        # 顶层模型
        self.top_model = TopModel()
    
    def forward(self, dense_input, sparse_input):
        # Guest方处理
        guest_output = self.guest_bottom(dense_input)
        # Host方处理
        host_output = self.host_bottom(sparse_input)
        # 顶层模型处理
        output = self.top_model(guest_output, host_output)
        
        return output


# 定义AUC评估函数
def auc(y_pred, y_true):
    # 获取预测值和真实值的数据
    pred = y_pred.data
    y = y_true.data
    # 计算并返回ROC AUC分数
    return roc_auc_score(y, pred)


# 定义绘制指标曲线的函数
def plot_metric(dfhistory, metric):
    # 获取训练集上的指标值
    train_metrics = dfhistory[metric]
    # 获取验证集上的指标值
    val_metrics = dfhistory['val_'+metric]
    # 创建epoch范围
    epochs = range(1, len(train_metrics) + 1)
    # 绘制训练集指标曲线
    plt.plot(epochs, train_metrics, 'bo--')
    # 绘制验证集指标曲线
    plt.plot(epochs, val_metrics, 'ro-')
    # 设置图表标题
    plt.title('Training and validation '+ metric)
    # 设置x轴标签
    plt.xlabel("Epochs")
    # 设置y轴标签
    plt.ylabel(metric)
    # 添加图例
    plt.legend(["train_"+metric, 'val_'+metric])
    # 显示图表
    plt.show()


# 定义训练函数
def train():
    # 准备数据
    fea_cols, (trn_dense, trn_sparse, trn_y), (val_dense, val_sparse, val_y) = prepared_data(file_path)

    # 把数据构建成数据管道
    # 创建训练集的TensorDataset
    dl_train_dataset = TensorDataset(
        torch.tensor(trn_dense).float(), 
        torch.tensor(trn_sparse).long(), 
        torch.tensor(trn_y).float()
    )
    # 创建验证集的TensorDataset
    dl_val_dataset = TensorDataset(
        torch.tensor(val_dense).float(), 
        torch.tensor(val_sparse).long(), 
        torch.tensor(val_y).float()
    )

    # 创建训练集的DataLoader
    dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=128)
    # 创建验证集的DataLoader
    dl_val = DataLoader(dl_val_dataset, shuffle=True, batch_size=128)

    # 看一下数据
    for b in iter(dl_train):
        # 打印第一个批次的形状和标签
        print(f"Dense: {b[0].shape}, Sparse: {b[1].shape}, Labels: {b[2].shape}")
        break

    # 创建VFL模型
    model = VFLModel(fea_cols)
    
    # 定义损失函数：二元交叉熵
    loss_func = nn.BCEWithLogitsLoss()
    # 定义优化器：Adam，学习率为0.001，添加权重衰减
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
    # 定义评估函数
    metric_func = auc
    # 定义评估指标名称
    metric_name = 'auc'
    # 设置训练轮数
    epochs = 10
    # 设置日志打印频率
    log_step_freq = 10

    # 创建DataFrame用于记录训练历史
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_'+metric_name])

    # 打印训练开始信息
    print('start_training.........')
    # 获取当前时间
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 打印分隔线和时间
    print('========'*8 + '%s' %nowtime)

    # 开始训练循环
    for epoch in range(1, epochs+1):
        
        # 训练阶段
        # 将模型设置为训练模式
        model.train()
        # 初始化损失和评估指标的累加值
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        # 遍历训练数据集
        for step, (dense_features, sparse_features, labels) in enumerate(dl_train, 1):
            # 梯度清零
            optimizer.zero_grad()
            
            # 正向传播
            # 获取模型预测
            outputs = model(dense_features, sparse_features)
            # 获取logits和概率
            logits = outputs["logits"].squeeze(-1)
            proba = outputs["proba"].squeeze(-1)
            # 计算损失
            loss = loss_func(logits, labels)
            try:
                # 计算评估指标
                metric = metric_func(proba, labels)
            except ValueError:
                # 处理可能的ValueError异常
                pass
            
            # 反向传播
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            
            # 打印batch级别日志
            # 累加损失
            loss_sum += loss.item()
            # 累加评估指标
            metric_sum += metric.item()
            # 按设定频率打印日志
            if step % log_step_freq == 0:
                print(f"[step={step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}")
        
        # 验证阶段
        # 将模型设置为评估模式
        model.eval()
        # 初始化验证损失和评估指标的累加值
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1
        
        # 遍历验证数据集
        for val_step, (dense_features, sparse_features, labels) in enumerate(dl_val, 1):
            # 不计算梯度
            with torch.no_grad():
                # 获取模型预测
                outputs = model(dense_features, sparse_features)
                # 获取logits和概率
                logits = outputs["logits"].squeeze(-1)
                proba = outputs["proba"].squeeze(-1)
                # 计算验证损失
                val_loss = loss_func(logits, labels)
                try:
                    # 计算验证评估指标
                    val_metric = metric_func(proba, labels)
                except ValueError:
                    # 处理可能的ValueError异常
                    pass
            
            # 累加验证损失
            val_loss_sum += val_loss.item()
            # 累加验证评估指标
            val_metric_sum += val_metric.item()
        
        # 记录日志
        # 创建包含当前epoch信息的元组
        info = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        # 将信息添加到历史记录中
        dfhistory.loc[epoch-1] = info
        
        # 打印日志
        # 打印当前epoch的训练和验证指标
        print(f"\nEPOCH={epoch}, loss={loss_sum/step:.3f}, {metric_name}={metric_sum/step:.3f}, val_loss={val_loss_sum/val_step:.3f}, val_{metric_name}={val_metric_sum/val_step:.3f}")
        # 获取当前时间
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 打印分隔线和时间
        print('\n' + '=========='* 8 + '%s' %nowtime)
    
    # 打印训练完成信息
    print('Finished Training')

    # 观察损失和准确率的变化
    # 绘制损失曲线
    plot_metric(dfhistory,"loss")
    # 绘制AUC曲线
    plot_metric(dfhistory,"auc")

    # 保存模型
    torch.save(model.state_dict(), 'vfl_model.pth')
    print("Model saved to vfl_model.pth")


# 主程序入口
if __name__ == '__main__':
    # 调用训练函数
    train()