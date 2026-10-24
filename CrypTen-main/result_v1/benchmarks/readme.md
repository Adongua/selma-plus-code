# Crypten运行效率测试

## 1 cpu文件夹下为使用cpu的测试，gpu文件夹下为使用gpu的测试

### 1.1 advanced_model文件夹存放带有较为先进的网络的测试结果，例如ResNet

### 1.2 img文件夹存放测试结果可视化效果图，img_e文件夹存放纵坐标为数量级形式的可视化结果
纵坐标为运行时间的中位数

### 1.3 multi_parties文件存放2~8方计算的测试结果

### 1.4 single_party文件存放单方计算的测试结果

#### 1.4.1 func_benchmarks.csv/func_benchmarks_cuda.csv为主要函数运行测试结果，时间的单位为s
主要函数分为一元运算函数（sigmoid、relu、tanh、exp、log、reciprocal、cos、sin、sum、mean、neg）、二元运算函数（add、sub、mul、matmul、gt、lt、eq）和层运算函数（conv2d、max_pool2d、avg_pool2d、dropout）。
- 参与运算的tensor的size为(100,100)
- conv2d的kernel的size为(10,10)
- pool2d的kernel的size为(3,3)
- dropout的dropout率为0.5
- 每个函数默认测试10次
- 列依次为函数名，pytorch运行时间的中位数、下四分位数、上四分位数，crypten运行时间的中位数、下四分位数、上四分位数，总绝对误差，平均相对误差

#### 1.4.2 func_benchmarks_comm.csv/func_benchmarks_comm_cuda.csv为主要函数通信开销测试结果，时间的单位为s
设置同func_benchmarks.csv/func_benchmarks_cuda.csv

#### 1.4.3 net_benchmarks_cuda.csv为pytorch与crypten的LeNet各层运行时间与通信开销测试结果，时间的单位为s
- LeNet结构：
  self.conv1 = nn.Conv2d(3,6,5)
  self.pool1 = nn.MaxPool2d(2,2)
  self.conv2 = nn.Conv2d(6,16,5)
  self.pool2 = nn.MaxPool2d(2,2)
  self.fc1 = nn.Linear(16*5*5,120)
  self.fc2 = nn.Linear(120,84)
  self.fc3 = nn.Linear(84,10)
- 数据集：CIFAR10（train=50000*3*32*32，test=10000*3*32*32）
- epochs = 10，batchsize = 100（但只训练1批次）

#### 1.4.4 model_benchmarks.csv为模型运行测试结果，时间的单位为s
主要模型为FNN、LR、ResNet
- 每个模型默认测试3次
- 每个模型出现的第一行为pytorch，第二行为crypten
- 列依次为模型名、训练一代所需时间的中位数、下四分位数、上四分位数，推理所需时间的中位数、下四分位数、上四分位数，是否明文
- FNN和LR的输入样本数量为5000，特征数量为20，通过sklearn的make_classification方法生成，分布为高斯簇，做二分类任务
- ResNet为torchvision中的预训练模型，输入样本一张图片调整成256\*256后中央裁取224\*224的图片，即输入的tensor的size为(1,3,224,224)（调整成满足ResNet的输入），做多分类任务
