![alt text](docs/public/dialogbot.jpg)

[![PyPI version](https://badge.fury.io/py/dialogbot.svg)](https://badge.fury.io/py/dialogbot)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Python3](https://img.shields.io/badge/Python-3.X-red.svg)

# dialogbot
dialogbot, provide complete dialogue model technology. Combining task-based dialogue model, search-based dialogue model and generative dialogue model, output the optimal dialogue response.

Base on **semantic analysis(deep learning)**, **knowledge graph(neo4j)** and **data mining(spider)**.

# Feature

### Retrieval Dialogue Bot

Compute questions similarity, use

- TFIDF
- BM25
- OneHot
- Query Vector

### Goal Oriented Dialogue Bot

- End to End Memory Networks(memn2n)
- BABi dataset

### Generative Dialogue Bot

- Sequence To Sequence Model(seq2seq)
- Taobao dataset


# Quick Start

## Requirements and Installation

The project is based on Tensorflow 1.12.0+ and Python 3.6+.
Then, simply do:

```
pip3 install dialogbot
```

or

```
git clone https://github.com/shibing624/dialogbot.git
cd dialogbot
python3 setup.py install
```

## Search Bot

Let's run chat bot:

```python
import dialogbot import Bot

bot = Bot()
response = bot.answer('亲 吃了吗？')
print(response)
```

Done!

This should print:

```
query: "亲 吃了吗？"

answer: "吃了的，你好呀"
```


## GPT2 Generation Bot
### Quick Run
在[模型分享](#模型分享)中下载模型，将模型文件夹model_epoch40_50w的文件放到项目根目录下，执行如下命令，进行对话
```
python interact.py --no_cuda --model_dir model_epoch40_50w (使用cpu生成，速度相对较慢)
或
python interact.py --model_dir model_epoch40_50w --device 0 (指定0号GPU进行生成，速度相对较快)
```

### 数据预处理
在项目根目录下创建data文件夹，将原始训练语料命名为train.txt，存放在该目录下。train.txt的格式如下，每段闲聊之间间隔一行，格式如下：
```
真想找你一起去看电影
突然很想你
我也很想你

想看你的美照
亲我一口就给你看
我亲两口
讨厌人家拿小拳拳捶你胸口

美女约嘛
开好房等你了
我来啦
```
运行preprocess.py，对data/train.txt对话语料进行tokenize，然后进行序列化保存到data/train.pkl。train.pkl中序列化的对象的类型为List[List],记录对话列表中,每个对话包含的token。
```
python preprocess.py --train_path data/train.txt --save_path data/train.pkl
```


### 训练模型
运行train.py,使用预处理后的数据，对模型进行自回归训练，模型保存在根目录下的model文件夹中。

在训练时，可以通过指定patience参数进行early stop。当patience=n时，若连续n个epoch，模型在验证集上的loss均没有下降，则进行early stop，停止训练。当patience=0时，不进行early stop。

代码中默认关闭了early stop，因为在实践中，early stop得到的模型的生成效果不一定会更好。
```
python train.py --epochs 40 --batch_size 8 --device 0,1 --train_path data/train.pkl
```
更多的训练参数介绍，可直接看train.py中的set_args()函数中的参数说明

### 预测模型（人机交互）
运行interact.py，使用训练好的模型，进行人机交互，输入q结束对话之后，聊天记录将保存到sample.txt文件中。
```
python interact.py --no_cuda --model_dir path_to_your_model
```
执行interact.py时，可以尝试通过调整topk、topp、repetition_penalty、max_history_len等参数，调整生成的效果。更多的参数介绍，可直接看interact.py的set_args()函数中的参数说明
如果要使用GPU进行生成，则不要调用--no_cuda参数，并且通过--device gpu_id来指定使用哪块GPU。


## 闲聊语料分享
|中文闲聊语料 | 数据集地址 |语料描述|
|---------|--------|--------|
|常见中文闲聊|[chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus)|包含小黄鸡语料、豆瓣语料、电视剧对白语料、贴吧论坛回帖语料、微博语料、PTT八卦语料、青云语料等|
|50w中文闲聊语料 | [百度网盘【提取码:4g5e】](https://pan.baidu.com/s/1M87Zf9e8iBqqmfTkKBWBWA) 或 [GoogleDrive](https://drive.google.com/drive/folders/1QFRsftLNTR_D3T55mS_FocPEZI7khdST?usp=sharing) |包含50w个多轮对话的原始语料、预处理数据|
|100w中文闲聊语料 | [百度网盘【提取码:s908】](https://pan.baidu.com/s/1TvCQgJWuOoK2f5D95nH3xg) 或 [GoogleDrive](https://drive.google.com/drive/folders/1NU4KLDRxdOGINwxoHGWfVOfP0wL05gyj?usp=sharing)|包含100w个多轮对话的原始语料、预处理数据|


中文闲聊语料的内容样例如下:
```
谢谢你所做的一切
你开心就好
开心
嗯因为你的心里只有学习
某某某，还有你
这个某某某用的好

你们宿舍都是这么厉害的人吗
眼睛特别搞笑这土也不好捏但就是觉得挺可爱
特别可爱啊

今天好点了吗？
一天比一天严重
吃药不管用，去打一针。别拖着
```

### 模型分享

|模型 | 共享地址 |模型描述|
|---------|--------|--------|
|model_epoch40_50w | [百度网盘(提取码:aisq)](https://pan.baidu.com/s/11KZ3hU2_a2MtI_StXBUKYw) 或 [GoogleDrive](https://drive.google.com/drive/folders/18TG2sKkHOZz8YlP5t1Qo_NqnGx9ogNay?usp=sharing) |使用50w多轮对话语料训练了40个epoch，loss降到2.0左右。|

## Contact

Please email your questions or comments to [xuming(shibing624)](http://www.borntowin.cn/).

## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
start with our [contributor guidelines](CONTRIBUTING.md) and then
check these [open issues](https://github.com/shibing624/dialogbot/issues) for specific tasks.

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit
tests for examples of how to call methods. Nearly all classes and methods are documented, so finding your way around
the code should hopefully be easy.


# Reference

- A Network-based End-to-End Trainable Task-oriented Dialogue System
Wen T H, Vandyke D, Mrksic N, et al. A Network-based End-to-End Trainable Task-oriented Dialogue System[J]. 2016.
当前构建一个诸如宾馆预订或技术支持服务的 task-oriented 的对话系统很难，主要是因为难以获取训练数据。现有两种方式解决问题：
•	将这个问题看做是 partially observable Markov Decision Process (POMDP)，利用强化学习在线与真实用户交互。但是语言理解和语言生成模块仍然需要语料去训练。而且为了让 RL 能运作起来，state 和 action space 必须小心设计，这就限制了模型的表达能力。同时 rewad function 很难设计，运行时也难以衡量
•	利用 seq2seq 来做，这又需要大量语料训练。同时，这类模型无法做到与数据库交互以及整合其他有用的信息，从而生成实用的相应。
本文提出了平衡两种方法的策略。
- How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation

- A. Bordes, Y. Boureau, J. Weston. Learning End-to-End Goal-Oriented Dialog 2016

- [chatbot-MemN2N-tensorflow](https://github.com/vyraun/chatbot-MemN2N-tensorflow)

- End-to-End Reinforcement Learning of Dialogue Agents for Information Access
2016年卡耐基梅隆大学研究团队利用深度强化学习进行对话状态追踪和管理

- Zhao T, Eskenazi M. Towards End-to-End Learning for Dialog State Tracking and Management using Deep Reinforcement Learning [J]. arXiv preprint arXiv:1606.02560, 2016.
2016年麻省理工大学研究团队提出层次化DQN模型，其代码采用Keras实现并开源[ code ]，该工作发表在NIPS2016上。

- Kulkarni T D, Narasimhan K R, Saeedi A, et al. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation [J]. arXiv preprint arXiv:1604.06057, 2016.

- BBQ-Networks: Efficient Exploration in Deep Reinforcement Learning for Task-Oriented Dialogue Systems
AAAI2018 录用文章，将深度强化学习用于对话系统。BBQ network 这个名字很有意思，工作来自微软研究院和 CMU。
提出了一种新的算法，可以显著提升对话系统中深度 Q 学习智能体的探索效率。我们的智能体通过汤普森采样（Thompson sampling）进行探索，可以从 Bayes-by-Backprop 神经网络中抽取蒙特卡洛样本。我们的算法的学习速度比 ε-greedy、波尔兹曼、bootstrapping 和基于内在奖励（intrinsic-reward）的方法等常用的探索策略快得多。此外，我们还表明：当 Q 学习可能失败时，只需将少数几个成功 episode 的经历叠加到重放缓冲（replay buffer）之上，就能使该 Q 学习可行。
- Deep Reinforcement Learning with Double Q-Learning

- Deep Attention Recurrent Q-Network

- SimpleDS: A Simple Deep Reinforcement Learning Dialogue System

- Deep Reinforcement Learning with a Natural Language Action Space

- Integrating User and Agent Models: A Deep Task-Oriented Dialogue System

- A Deep Reinforcement Learning Chatbot
蒙特利尔算法研究实验室（MILA）为参与亚马逊 Alexa 大奖赛而开发的深度强化学习聊天机器人。
MILABOT 能够与人类就流行的闲聊话题进行语音和文本交流。该系统包括一系列自然语言生成和检索模型，如模板模型、词袋模型、序列到序列神经网络和隐变量神经网络模型。
通过将强化学习应用到众包数据和真实用户互动中进行训练，该系统学习从自身包含的一系列模型中选择合适的模型作为响应。
真实用户使用 A/B 测试对该系统进行评估，其性能大大优于竞争系统。由于其机器学习架构，该系统的性能在额外数据的帮助下还有可能继续提升。


# License

[The Apache License 2.0](/LICENSE)