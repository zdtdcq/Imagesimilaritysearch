1 引入相应的库
=========


```python
import paddle
import paddle.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


2 数据加载
====
	 本示例采用CIFAR-10数据集。这是一个经典的数据集，由50000张图片的训练数据，和10000张图片的测试数据组成，其中每张图片是一个RGB的长和宽都为32的图片。使用paddle.vision.datasets.Cifar10可以方便的完成数据的下载工作，把数据归一化到(0, 1.0)区间内，并提供迭代器供按顺序访问数据。将训练数据和测试数据分别存放在两个numpy数组中，供后面的训练和评估来使用。



```python
import paddle.vision.transforms as T
transform = T.Compose([T.Transpose((2, 0, 1))])

cifar10_train = paddle.vision.datasets.Cifar10(mode='train', transform=transform)
x_train = np.zeros((50000, 3, 32, 32))
y_train = np.zeros((50000, 1), dtype='int32')

for i in range(len(cifar10_train)):
    train_image, train_label = cifar10_train[i]
    
    # normalize the data
    x_train[i,:, :, :] = train_image / 255.
    y_train[i, 0] = train_label

y_train = np.squeeze(y_train)

print(x_train.shape)
print(y_train.shape)

cifar10_test = paddle.vision.datasets.cifar.Cifar10(mode='test', transform=transform)
x_test = np.zeros((10000, 3, 32, 32), dtype='float32')
y_test = np.zeros((10000, 1), dtype='int64')

for i in range(len(cifar10_test)):
    test_image, test_label = cifar10_test[i]
   
    # normalize the data
    x_test[i,:, :, :] = test_image / 255.
    y_test[i, 0] = test_label

y_test = np.squeeze(y_test)

print(x_test.shape)
print(y_test.shape)

```

    item     8/41626 [..............................] - ETA: 2:59 - 4ms/ite

    Cache file /home/aistudio/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz not found, downloading https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz 
    Begin to download


    item    81/41626 [..............................] - ETA: 3:17 - 5ms/itemitem   907/41626 [..............................] - ETA: 1:04 - 2ms/
    item  3026/41626 [=>............................] - ETA: 36s - 935us/it

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item  4821/41626 [==>...........................] - ETA: 30s - 822us/i
    item  6957/41626 [====>.........................] - ETA: 26s - 763us/ite

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item  8635/41626 [=====>........................] - ETA: 24s - 740us/it
    item 10951/41626 [======>.......................] - ETA: 22s - 721us/it

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item 12108/41626 [=======>......................] - ETA: 21s - 715us/it
    item 14815/41626 [=========>....................] - ETA: 18s - 708us/item

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item 16625/41626 [==========>...................] - ETA: 17s - 701us/it
    item 18825/41626 [============>.................] - ETA: 15s - 695us/it

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item 20059/41626 [=============>................] - ETA: 14s - 694us/it
    item 22755/41626 [===============>..............] - ETA: 13s - 692us/it

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item 23781/41626 [================>.............] - ETA: 12s - 690us/ite
    item 26519/41626 [==================>...........] - ETA: 10s - 686us/ite

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item 28276/41626 [===================>..........] - ETA: 9s - 688us/
    item 30377/41626 [====================>.........] - ETA: 7s - 684us/it

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item 31506/41626 [=====================>........] - ETA: 6s - 683us/
    item 34024/41626 [=======================>......] - ETA: 5s - 688us/item

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item 35461/41626 [========================>.....] - ETA: 4s - 687us/
    item 37896/41626 [==========================>...] - ETA: 2s - 693us/it

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    item 38950/41626 [===========================>..] - ETA: 1s - 694us/it
    item 41626/41626 [============================>.] - ETA: 0s - 694us/it

    
    Download finished


    (50000, 3, 32, 32)
    (50000,)
    (10000, 3, 32, 32)
    (10000,)


2.2 数据探索
----
接下来随机从训练数据里找一些图片，浏览一下这些图片。


```python
height_width = 32

def show_collage(examples):
    box_size = height_width + 2
    num_rows, num_cols = examples.shape[:2]

    collage = Image.new(
        mode="RGB",
        size=(num_cols * box_size, num_rows * box_size),
        color=(255, 255, 255),
    )
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            array = (np.array(examples[row_idx, col_idx]) * 255).astype(np.uint8)
            array = array.transpose(1,2,0)
            collage.paste(
                Image.fromarray(array), (col_idx * box_size, row_idx * box_size)
            )

    collage = collage.resize((2 * num_cols * box_size, 2 * num_rows * box_size))
    return collage

sample_idxs = np.random.randint(0, 50000, size=(5, 5))
examples = x_train[sample_idxs]
show_collage(examples)

```




![png](output_5_0.png)



3 构建训练数据
===
	图片检索的模型的训练样本跟常见的分类任务的训练样本不太一样的地方在于，每个训练样本并不是一个(image, class)这样的形式。而是（image0, image1,similary_or_not)的形式，即，每一个训练样本由两张图片组成，而其label是这两张图片是否相似的标志位（0或者1）。很自然的能够想到，来自同一个类别的两张图片，是相似的图片，而来自不同类别的两张图片，应该是不相似的图片。为了能够方便的抽样出相似图片（以及不相似图片）的样本，先建立能够根据类别找到该类别下所有图片的索引。


```python
class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)

```

**有了上面的索引，就可以为飞桨准备一个读取数据的迭代器。该迭代器每次生成2 * number of classes张图片，在CIFAR10数据集中，这会是20张图片。前10张图片，和后10张图片，分别是10个类别中每个类别随机抽出的一张图片。这样，在实际的训练过程中，就会有10张相似的图片和90张不相似的图片（前10张图片中的任意一张图片，都与后10张的对应位置的1张图片相似，而与其他9张图片不相似）。**



```python
num_classes = 10

def reader_creator(num_batchs):
    def reader():
        iter_step = 0
        while True:
            if iter_step >= num_batchs:
                break
            iter_step += 1
            x = np.empty((2, num_classes, 3, height_width, height_width), dtype=np.float32)
            for class_idx in range(num_classes):
                examples_for_class = class_idx_to_train_idxs[class_idx]
                anchor_idx = random.choice(examples_for_class)
                positive_idx = random.choice(examples_for_class)
                while positive_idx == anchor_idx:
                    positive_idx = random.choice(examples_for_class)
                x[0, class_idx] = x_train[anchor_idx]
                x[1, class_idx] = x_train[positive_idx]
            yield x

    return reader


# num_batchs: how many batchs to generate
def anchor_positive_pairs(num_batchs=100):
    return reader_creator(num_batchs)
pairs_train_reader = anchor_positive_pairs(num_batchs=1000)


```

**拿出第一批次的图片，并可视化的展示出来，如下所示。**


```python
examples = next(pairs_train_reader())
print(examples.shape)
show_collage(examples)
```

    (2, 10, 3, 32, 32)





![png](output_11_1.png)



4 模型组网
==
把图片转换为高维的向量表示的网络
---


```python
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(in_channels=3, 
                                      out_channels=32, 
                                      kernel_size=(3, 3),
                                      stride=2)
         
        self.conv2 = paddle.nn.Conv2D(in_channels=32, 
                                      out_channels=64, 
                                      kernel_size=(3,3), 
                                      stride=2)       
        
        self.conv3 = paddle.nn.Conv2D(in_channels=64, 
                                      out_channels=128, 
                                      kernel_size=(3,3),
                                      stride=2)
       
        self.gloabl_pool = paddle.nn.AdaptiveAvgPool2D((1,1))

        self.fc1 = paddle.nn.Linear(in_features=128, out_features=8)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.gloabl_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self.fc1(x)
        x = x / paddle.norm(x, axis=1, keepdim=True)
        return x

```

5 模型训练
==
- inverse_temperature参数起到的作用是让softmax在计算梯度时，能够处于梯度更显著的区域。（可以参考attention is all you need中，在点积之后的scale操作）。

- 整个计算过程，会先用上面的网络分别计算前10张图片（anchors)的高维表示，和后10张图片的高维表示。然后再用matmul计算前10张图片分别与后10张图片的相似度。（所以similarities会是一个(10, 10)的Tensor）。

- 在构造类别标签时，则相应的，可以构造出来0 ~ num_classes的标签值，用来让学习的目标成为相似的图片的相似度尽可能的趋向于1.0，而不相似的图片的相似度尽可能的趋向于-1.0。


```python
def train(model):
    print('start training ... ')
    model.train()

    inverse_temperature = paddle.to_tensor(np.array([1.0/0.2], dtype='float32'))

    epoch_num = 20
    
    opt = paddle.optimizer.Adam(learning_rate=0.0001,
                                parameters=model.parameters())
    
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(pairs_train_reader()):
            anchors_data, positives_data = data[0], data[1]

            anchors = paddle.to_tensor(anchors_data)
            positives = paddle.to_tensor(positives_data)
            
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            
            similarities = paddle.matmul(anchor_embeddings, positive_embeddings, transpose_y=True) 
            similarities = paddle.multiply(similarities, inverse_temperature)
            
            sparse_labels = paddle.arange(0, num_classes, dtype='int64')

            loss = F.cross_entropy(similarities, sparse_labels)
            
            if batch_id % 500 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()

model = MyNet()
train(model)

```

    W0224 11:03:22.013046   101 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0224 11:03:22.018281   101 device_context.cc:465] device: 0, cuDNN Version: 7.6.


    start training ... 
    epoch: 0, batch_id: 0, loss is: [2.2612557]
    epoch: 0, batch_id: 500, loss is: [2.1549609]
    epoch: 1, batch_id: 0, loss is: [2.033414]
    epoch: 1, batch_id: 500, loss is: [2.3011215]
    epoch: 2, batch_id: 0, loss is: [2.0282888]
    epoch: 2, batch_id: 500, loss is: [1.9078135]
    epoch: 3, batch_id: 0, loss is: [1.7453687]
    epoch: 3, batch_id: 500, loss is: [1.9648392]
    epoch: 4, batch_id: 0, loss is: [1.9464357]
    epoch: 4, batch_id: 500, loss is: [1.7301369]
    epoch: 5, batch_id: 0, loss is: [1.7679822]
    epoch: 5, batch_id: 500, loss is: [1.6645566]
    epoch: 6, batch_id: 0, loss is: [2.0526464]
    epoch: 6, batch_id: 500, loss is: [1.6891029]
    epoch: 7, batch_id: 0, loss is: [1.8070191]
    epoch: 7, batch_id: 500, loss is: [1.9409759]
    epoch: 8, batch_id: 0, loss is: [1.8104442]
    epoch: 8, batch_id: 500, loss is: [1.6778669]
    epoch: 9, batch_id: 0, loss is: [1.8782021]
    epoch: 9, batch_id: 500, loss is: [2.2547379]
    epoch: 10, batch_id: 0, loss is: [1.9428344]
    epoch: 10, batch_id: 500, loss is: [1.6480865]
    epoch: 11, batch_id: 0, loss is: [1.9377271]
    epoch: 11, batch_id: 500, loss is: [2.0245948]
    epoch: 12, batch_id: 0, loss is: [1.8119581]
    epoch: 12, batch_id: 500, loss is: [1.9946213]
    epoch: 13, batch_id: 0, loss is: [1.8884126]
    epoch: 13, batch_id: 500, loss is: [2.1792078]
    epoch: 14, batch_id: 0, loss is: [1.7868779]
    epoch: 14, batch_id: 500, loss is: [1.7087588]
    epoch: 15, batch_id: 0, loss is: [1.8030777]
    epoch: 15, batch_id: 500, loss is: [1.9514712]
    epoch: 16, batch_id: 0, loss is: [1.7788224]
    epoch: 16, batch_id: 500, loss is: [1.4080791]
    epoch: 17, batch_id: 0, loss is: [1.8306112]
    epoch: 17, batch_id: 500, loss is: [2.0306332]
    epoch: 18, batch_id: 0, loss is: [2.1214547]
    epoch: 18, batch_id: 500, loss is: [2.103322]
    epoch: 19, batch_id: 0, loss is: [1.8716949]
    epoch: 19, batch_id: 500, loss is: [1.490519]


6 模型预测
==
- 前述的模型训练训练结束之后，就可以用该网络结构来计算出任意一张图片的高维向量表示（embedding)，通过计算该图片与图片库中其他图片的高维向量表示之间的相似度，就可以按照相似程度进行排序，排序越靠前，则相似程度越高。

- 下面对测试集中所有的图片都两两计算相似度，然后选一部分相似的图片展示出来。


```python
near_neighbours_per_example = 10

x_test_t = paddle.to_tensor(x_test)
test_images_embeddings = model(x_test_t)
similarities_matrix = paddle.matmul(test_images_embeddings, test_images_embeddings, transpose_y=True) 

indicies = paddle.argsort(similarities_matrix, descending=True)
indicies = indicies.numpy()
examples = np.empty(
    (
        num_classes,
        near_neighbours_per_example + 1,
        3,
        height_width,
        height_width,
    ),
    dtype=np.float32,
)

for row_idx in range(num_classes):
    examples_for_class = class_idx_to_test_idxs[row_idx]
    anchor_idx = random.choice(examples_for_class)
    
    examples[row_idx, 0] = x_test[anchor_idx]
    anchor_near_neighbours = indicies[anchor_idx][1:near_neighbours_per_example+1]
    for col_idx, nn_idx in enumerate(anchor_near_neighbours):
        examples[row_idx, col_idx + 1] = x_test[nn_idx]

show_collage(examples)

```




![png](output_17_0.png)




```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
# !ls /home/aistudio/data
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# # Please clean unnecessary files in time to speed up environment loading. 
# !ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# # you need to use the persistence path as the following: 
# !mkdir /home/aistudio/external-libraries
# !pip install beautifulsoup4 -t /home/aistudio/external-libraries
```


```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
# import sys 
# sys.path.append('/home/aistudio/external-libraries')
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
