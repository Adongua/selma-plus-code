import math
import numpy as np
# from transformers import BertForMaskedLM
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
from prv_accountant import Accountant
import crypten


# Matrix Gaussian Noise
def matrix_gaussian_noise(epsilon, delta, sensitivity):
    """
    根据给定的隐私参数epsilon、delta和敏感性sensitivity，生成相应的矩阵高斯噪声。
    
    参数:
    epsilon -- 隐私保护参数epsilon，表示允许的信息泄露量。
    delta -- 隐私保护参数delta，表示发生特定泄露事件的概率。
    sensitivity -- 输入数据的敏感性，即任意两个相邻输入之间的差值的范数。
    
    返回:
    noise_b -- 生成的噪声的标量，用于生成高斯噪声矩阵。
    """
    def function_phi(t):
        """
        计算标准正态分布的累积分布函数值。
        
        参数:
        t -- 输入值。
        
        返回:
        function_phi(t) -- t处的标准正态分布的累积分布函数值。
        """
        return (1 + math.erf(t / math.sqrt(2))) / 2

    def B_plus_function(v, epsilon):
        """
        计算B+函数的值。
        
        参数:
        v -- 输入值。
        epsilon -- 隐私保护参数epsilon。
        
        返回:
        B_plus_function(v, epsilon) -- v和epsilon处的B+函数值。
        """
        return function_phi(math.sqrt(epsilon * v)) - math.exp(epsilon) * function_phi(-math.sqrt(epsilon * (v + 2)))

    def B_minus_function(u, epsilon):
        """
        计算B-函数的值。
        
        参数:
        u -- 输入值。
        epsilon -- 隐私保护参数epsilon。
        
        返回:
        B_minus_function(u, epsilon) -- u和epsilon处的B-函数值。
        """
        return function_phi(-math.sqrt(epsilon * u)) - math.exp(epsilon) * function_phi(-math.sqrt(epsilon * (u + 2)))

    def compute_R(epsilon, delta, iterations=5000):
        """
        计算R值，用于确定噪声的规模。
        
        参数:
        epsilon -- 隐私保护参数epsilon。
        delta -- 隐私保护参数delta。
        iterations -- 二分查找的迭代次数，默认为5000。
        
        返回:
        R -- 根据epsilon和delta计算得到的噪声规模调整因子。
        """
        delta_0 = function_phi(0) - math.exp(epsilon) * function_phi(-math.sqrt(2 * epsilon))
        # print(delta_0)

        start, end = 0, 1e5

        B_function = B_plus_function if delta >= delta_0 else B_minus_function
        for i in range(iterations):
            mid = (start + end) / 2
            value = B_function(mid, epsilon)
            if value < delta:
                end = mid
            else:
                start = mid

        u_star = end
        b_value = B_function(end, epsilon)

        if delta >= delta_0:
            alpha = math.sqrt(1 + u_star / 2) - math.sqrt(u_star / 2)
        else:
            alpha = math.sqrt(1 + u_star / 2) + math.sqrt(u_star / 2)

        R = math.sqrt(2 * epsilon) / alpha

        return R

    R = compute_R(epsilon, delta)
    noise_b = sensitivity / R
    # noise_matrix = noise_b * np.random.normal(size=dim)

    return noise_b


# https://en.m.wikipedia.org/wiki/Additive_noise_mechanisms#Laplace_Mechanism
def Lap_noise(epsilon, dim, sensitivity):
    """
    根据Laplace机制生成加性噪声。
  
    参数:
    - epsilon (float): 隐私预算，表示允许的最大隐私泄露量。
    - dim (int): 噪声向量的维度，即需要添加噪声的数据的维度。
    - sensitivity (float): 函数的敏感度，表示输出结果变化1个单位时，输入变化的最大值。

    返回:
    - noise_matrix (np.ndarray): 生成的噪声矩阵，维度为dim，用于添加到数据中以保护隐私。
    """
    # 根据隐私预算和敏感度计算噪声强度
    noise_b = sensitivity / epsilon
    noise_matrix = np.random.laplace(loc=0, scale=noise_b, size=dim)
    return noise_matrix


# https://en.m.wikipedia.org/wiki/Additive_noise_mechanisms#Gaussian_Mechanism
def Gaussian_noise(epsilon, delta, sensitivity):
    """
    生成符合高斯分布的噪声，用于添加到数据中以实现差分隐私保护。

    参数:
    epsilon (float): 隐私预算，控制噪声的添加量。
    delta (float): 隐私参数，指定一个非常小的概率，函数保证在该概率下满足差分隐私。
    sensitivity (float): 输入数据的敏感度，即最大变化值。

    返回:
    noise_b (float): 根据高斯机制计算得到的噪声值。
    """
    # 根据高斯机制计算噪声的标准差
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    noise_b = sensitivity * sigma
    # noise_matrix = np.random.normal(loc=0,scale=noise_b,size=dim)
    return noise_b


# MVG@CCS18
def MVG_noise(epsilon, delta, dim, sensitivity, gamma):
    """
    根据给定的隐私参数和数据特性，计算添加到数据上的高斯噪声量。
    
    参数:
    epsilon - 隐私保护的epsilon值，控制噪声的添加量。
    delta - 隐私保护的delta值，表示在给定epsilon的情况下，不满足隐私定义的概率。
    dim - 输入数据的维度，以元组形式表示。
    sensitivity - 输入数据的敏感性，即任意两个相邻数据之间的差异。
    gamma - 用于调整噪声分布的参数。
    
    返回:
    noise_b - 根据给定参数计算得到的高斯噪声的标量。
    """
    def cal_harmonic_number(r, m=1.0):
        # 计算r的调和数
        sum = 0
        for i in range(1, r + 1):
            sum += 1 / i ** m
        return sum

    r = min(dim) # 取维度中的较小值
    H_r = cal_harmonic_number(r) # 计算r的整数调和数
    H_r_half = cal_harmonic_number(r, m=0.5) # 计算r的0.5次调和数
    alpha = (H_r + H_r_half) * gamma ** 2 + 2 * H_r * gamma * sensitivity
    zeta = 2 * math.sqrt(-dim[0] * dim[1] * math.log(delta)) - 2 * math.log(delta) + dim[0] * dim[1]
    beta = 2 * ((dim[0] * dim[1]) ** 0.25) * H_r * sensitivity * zeta
    noise_b = (2 * alpha * ((dim[0] * dim[1]) ** 0.25)) / (-beta + math.sqrt(beta ** 2 + 8 * alpha * epsilon))
    #
    # precisionBudget = ((-beta + np.sqrt(beta ** 2 + 8 * alpha * epsilon)) ** 2) / (4 * (alpha ** 2))
    return noise_b


def gamma_simulation(num_simulation=100000):
    """
    执行BERT嵌入向量的gamma模拟。

    参数:
    num_simulation (int): 模拟的次数，默认为100,000次。

    返回:
    float: 所有模拟中最大的范数值。
    """
    # 加载预训练的BERT模型
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    # 获取BERT嵌入层的权重矩阵
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    total_num = embedding_matrix.shape[0] # 嵌入向量的总数
    total_indices = np.arange(total_num) # 生成嵌入向量的索引范围
    norms = [] # 存储每次模拟的范数结果
    # 对于每次模拟，随机重排索引并计算选取的嵌入向量的范数
    for i in tqdm(range(num_simulation)):
        np.random.shuffle(total_indices) # 随机洗牌索引
        sampled_indices = total_indices[:512] # 随机采样512个索引
        embeddings = embedding_matrix[sampled_indices] # 获取对应的嵌入向量
        norm2 = np.linalg.norm(embeddings) # 计算嵌入向量的范数
        norms.append(norm2) # 将范数结果添加到列表中
    print(max(norms))
    return max(norms)


def cal_sensitivity_bert_embedding():
    """
    This function may run up to a few minutes. The results are:
    L1 Sensitivity: 56.33
    L2 Sensitivity: 2.89

    计算BERT嵌入的敏感性。
    该函数可能运行几分钟。结果为：
    L1 敏感性：56.33
    L2 敏感性：2.89
    
    参数: 
    无
    
    返回:
    tuple: 包含L1和L2敏感性的元组。
    """
 
    # 加载预训练的BERT模型
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    # 获取BERT嵌入层的嵌入矩阵
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    # l1-sensitivity
    distance = cdist(embedding_matrix, embedding_matrix, "cityblock")
    l1_sensitivity = np.max(distance)
    print(l1_sensitivity)

    # l2-sensitivity
    distance = cdist(embedding_matrix, embedding_matrix, "euclidean")
    l2_sensitivity = np.max(distance)
    print(l2_sensitivity)

    return l1_sensitivity, l2_sensitivity


def get_noise_multiplier(eps, delta, batch_size=1, dataset_size=50000, epoch=3, local_dp=False, noise_type='aGM'):
    """
    根据给定的参数计算噪声乘数。

    参数:
    - eps: 隐私预算epsilon，用于控制DP算法的隐私泄露程度。
    - delta: 隐私预算delta，用于控制在epsilon固定的情况下，更精确地调整隐私泄露的概率。
    - batch_size: 批量处理的样本数量，默认为1。
    - dataset_size: 数据集的总样本数量，默认为50000。
    - epoch: 训练的轮数，默认为3。
    - local_dp: 是否使用本地差分隐私，默认为False。
    - noise_type: 噪声类型，'aGM'代表加权矩阵高斯噪声，'GM'代表高斯噪声，默认为'aGM'。

    返回值:
    - 计算得到的噪声乘数，用于在给定隐私预算下调整噪声的强度。
    """
    if local_dp:
        if noise_type == 'aGM':
            return matrix_gaussian_noise(epsilon=eps,delta=delta,sensitivity=1)
        elif noise_type == 'GM':
            return Gaussian_noise(epsilon=eps,delta=delta,sensitivity=1)

    start_noise_multiplier = 0.2
    end_noise_multiplier = 50
    while True:
        mid_noise_multiplier = (start_noise_multiplier + end_noise_multiplier) / 2
        accountant = Accountant(
            noise_multiplier=mid_noise_multiplier,
            sampling_probability=batch_size / dataset_size,
            delta=1e-5,
            eps_error=0.1,
            max_compositions=round(dataset_size / batch_size) * epoch,
        )
        _, eps_estimate, _ = accountant.compute_epsilon(
            num_compositions=round(dataset_size / batch_size) * epoch)
        # print(eps_estimate)
        if abs(eps_estimate - eps) < 0.0001:
            break
        # less noise
        if eps_estimate > eps:
            start_noise_multiplier = mid_noise_multiplier
        else:
            end_noise_multiplier = mid_noise_multiplier

    return mid_noise_multiplier



def add_noise(embeddings,noise_factor=0.5,norm_c=1.0,add_noise=True):
    """
    向嵌入向量中添加噪声。
    
    参数:
    - embeddings: 原始嵌入向量，要求是一个张量。
    - noise_factor: 添加的噪声的强度因子，默认为0.5。
    - norm_c: 对嵌入向量进行标准化的常数，默认为1.0。
    - add_noise: 一个布尔值，控制是否添加噪声，默认为True。
    
    返回:
    - 添加了噪声的嵌入向量。
    """
    if add_noise:
        # 对嵌入向量进行最大范数标准化
        embeddings = _max_norm_clip(embeddings, norm_c)
        # 生成与嵌入向量相同形状的高斯噪声
        noise_embeds = noise_factor * crypten.randn(embeddings.shape)
        noise_embeds = noise_embeds.to(embeddings.device)
        # 将噪声加到嵌入向量上
        embeddings = noise_embeds + embeddings
    return embeddings


def _max_norm_clip(embeddings, norm_c=1.0):
    """
    对嵌入向量进行最大范数裁剪。

    参数:
    embeddings - 输入的嵌入张量。
    norm_c - 裁剪的范数阈值，默认为1.0。

    返回:
    裁剪后的嵌入张量。
    """
    shape = embeddings.shape
    embeddings = embeddings.reshape(shape[0], -1)
    total_norm = embeddings.norm(dim=-1) # 很耗时
    # print(total_norm.mean())
    total_norm_rec = (total_norm + 1e-6).reciprocal();
    clip_coef = norm_c * total_norm_rec
    # 使用 minimum 替代 clamp
    clip_coef_clamped = clip_coef.where(clip_coef > 1.0, torch.tensor(1.0))
    embeddings = embeddings.mul(clip_coef_clamped.unsqueeze(-1))
    embeddings = embeddings.reshape(shape)

    return embeddings
