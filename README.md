![alt text](docs/public/dialogbot.jpg)

[![PyPI version](https://badge.fury.io/py/dialogbot.svg)](https://badge.fury.io/py/dialogbot)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/deepmipt/DeepPavlov/blob/master/LICENSE)
![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Python3](https://img.shields.io/badge/Python-3.X-red.svg)

# dialogbot
dialogbot, provide complete dialogue model technology. Combining **search-based dialogue model**, **task-based dialogue model** and **generative dialogue model**, output the optimal dialogue response.

对话机器人，基于问答型对话，任务型对话，聊天型对话等模型实现，支持网络检索问答，领域知识问答，任务引导问答，闲聊问答，开箱即用。



**Guide**

- [Question](#Question)
- [Solution](#Solution)
- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Dataset](#Dataset)
- [Contact](#Contact)
- [Cite](#Cite)
- [Reference](#reference)

# Question

人机对话系统一直是AI的重要方向，图灵测试以对话检测机器是否拥有高度的智能。

如何构建人机对话系统或者对话机器人呢？


# Solution

对话系统经过三代的演变：

1. 规则对话系统：垂直领域可以利用模板匹配方法的匹配问句和相应的答案。优点是内部逻辑透明，易于分析调试，缺点是高度依赖专家干预，
缺少灵活性和可可拓展性。
2. 统计对话系统：基于部分可见马尔科夫决策过程的统计对话系统，先对问句进行贝叶斯推断，维护每轮对话状态，再跟进对话状态进行对话策略的选择，
从而生成自然语言回复。基本形成现代的对话系统框架，它避免了对专家的高度依赖，缺点是模型难以维护，可拓展性比较受限。
3. 深度对话系统：基本延续了统计对话系统的框架，但各个模型采用深度网络模型。利用了深度模型强大的表征能力，语言分类和生成能力大幅提高，
缺点是需要大量标注数据才能有效训练模型。

对话系统分为三类：

- 问答型对话：多是一问一答，用户提问，系统通过对问题解析和查找知识库返回正确答案，如搜索。
- 任务型对话：指由任务驱动的多轮对话，机器需要通过理解、主动询问、澄清等方式确定用户目标，然后查找知识库返回结果，完成用户需求。
如：机器人售电影票。
- 聊天型对话：目标是产生有趣且富有信息量的自然答复使人机对话持续下去，如小度音响。


# Feature

### 问答型对话（Search Dialogue Bot）

#### 本地检索问答

计算用户问句与问答库中问句的相似度，选择最相似的问句，给出其对应的答复。

句子相似度计算包括以下方法：

- TFIDF
- BM25
- OneHot
- Query Vector

#### 网络检索问答

对百度、Bing的搜索结果摘要进行答案的检索
- 百度搜索，包括百度知识图谱、百度诗词、百度万年历、百度计算器、百度知道
- 微软Bing搜索，包括bing知识图谱、bing网典


### 任务型对话（Task Oriented Dialogue Bot）

- End to End Memory Networks(memn2n)
- BABi dataset

### 聊天型对话（Generative Dialogue Bot）

- GPT2 Model
- Sequence To Sequence Model(seq2seq)
- Taobao dataset


# Install

- Requirements and Installation

The project is based on transformers 4.4.2+, torch 1.6.0+ and Python 3.6+.
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

# Usage
## 问答型对话（Search Bot）

示例[base_demo.py](examples/base_demo.py)

```python
import dialogbot import Bot

bot = Bot()
response = bot.answer('姚明多高呀？')
print(response)
```

output:

```
query: "姚明多高呀？"

answer: "226cm"
```

## 任务型对话（Task Bot）

示例[taskbot_demo.py](examples/taskbot_demo.py)






## 聊天型对话（Generative Bot）

### GPT2模型使用
基于GPT2生成模型训练的聊天型对话模型。

在[模型分享](#模型分享)中下载模型，将模型文件夹model_epoch40_50w下的文件放到自己指定目录`your_model_dir`下：
```
model_epoch40_50w
├── config.json
├── pytorch_model.bin
└── vocab.txt
```

示例[genbot_demo.py](examples/genbot_demo.py)


```python
import dialogbot import Bot

bot = Bot(gpt_model_path=your_model_dir)
response = bot.answer('亲 你吃了吗？', use_gen=True, use_search=False, use_task=False)
print(response)
```

output:

```
query: "亲 吃了吗？"

answer: "吃了"
```


### GPT2模型fine-tune


#### 数据预处理
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
cd dialogbot/gpt/
python preprocess.py --train_path data/train.txt --save_path data/train.pkl
```


#### 训练模型
运行train.py,使用预处理后的数据，对模型进行自回归训练，模型保存在根目录下的model文件夹中。

在训练时，可以通过指定patience参数进行early stop。当patience=n时，若连续n个epoch，模型在验证集上的loss均没有下降，则进行early stop，停止训练。当patience=0时，不进行early stop。

代码中默认关闭了early stop，因为在实践中，early stop得到的模型的生成效果不一定会更好。
```
python train.py --epochs 40 --batch_size 8 --device 0,1 --train_path data/train.pkl
```
更多的训练参数介绍，可直接看train.py中的set_args()函数中的参数说明

#### 预测模型（人机交互）
运行interact.py，使用训练好的模型，进行人机交互，输入q结束对话之后，聊天记录将保存到sample.txt文件中。
```
python interact.py --no_cuda --model_dir path_to_your_model
```
执行interact.py时，可以尝试通过调整topk、topp、repetition_penalty、max_history_len等参数，调整生成的效果。更多的参数介绍，可直接看interact.py的set_args()函数中的参数说明
如果要使用GPU进行生成，则不要调用--no_cuda参数，并且通过--device gpu_id来指定使用哪块GPU。


##### 问答示例
- sample1
```
  user :在干嘛
chatbot:睡觉
  user :这么早就睡觉啦
chatbot:嗯啊你也早点睡啊
  user :我再玩一会儿手机
chatbot:早点休息晚安
  user :拜拜啦
chatbot:晚安晚安
  user :多喝热水
chatbot:哈哈哈
  user :多喝热水
chatbot:好的咯
```

- sample2
```
  user :在吗
chatbot:在
  user :我不开心
chatbot:怎么啦
  user :最近好累啊
chatbot:是挺累的
  user :作业好多啊
chatbot:还好还好
  user :我觉得我快要挂科了
chatbot:我也是啊，但是我觉得我快熬出头了
  user :不说了，我还有好多作业没写
chatbot:我也是啊好多啊
  user :那先这样，下次再聊，再见了
chatbot:好哒
```

# Dataset

### 闲聊语料分享
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


# Contact

- 邮件我：xuming: shibing624@126.com.
- 微信我：
加我*微信号：xuming624, 备注：个人名称-NLP* 进NLP交流群。

<img src="docs/public/wechat.jpeg" width="200" />


# Cite

如果你在研究中使用了dialogbot，请按如下格式引用：

```latex
@software{dialogbot,
  author = {Xu Ming},
  title = {dialogbot: Dialogue Model Technology Tool},
  year = {2021},
  url = {https://github.com/shibing624/dialogbot},
}
```

# License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加dialogbot的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python setup.py test`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。


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

- [The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- [transformers](https://github.com/huggingface/transformers)
- [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
- [DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.xilesou.top/pdf/1911.00536.pdf)
- [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
