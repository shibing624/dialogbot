# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Modified on: https://github.com/yangjianxin1/GPT2-chitchat
"""
import argparse
import os
import pickle
import sys
from datetime import datetime
from os.path import join
from loguru import logger
from tqdm.auto import tqdm, trange
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import transformers
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Config

sys.path.append('../..')
from dialogbot.gpt.earlystop import EarlyStopping


class MyDataset(Dataset):
    """
    GPT model dataset
    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def load_dataset(args):
    """
    加载训练集和验证集
    """
    logger.info("loading training dataset and validating dataset")
    train_path = args.train_path

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    # 划分训练集与验证集
    val_num = int(len(input_list) * args.val_rate)
    logger.info(f'data size: {len(input_list)}, val_num: {val_num}')
    input_list_train = input_list[val_num:]
    input_list_val = input_list[:val_num]

    train_dataset = MyDataset(input_list_train, args.max_len)
    val_dataset = MyDataset(input_list_val, args.max_len)

    return train_dataset, val_dataset


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch, args):
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss的总和

    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0

    batch_iterator = tqdm(
        train_dataloader,
        desc=f"Running Epoch {epoch} of {args.epochs}",
        mininterval=0,
    )
    step_num = len(batch_iterator)
    for batch_idx, (input_ids, labels) in enumerate(batch_iterator):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                batch_iterator.set_description(
                    f"Epochs {epoch + 1}/{args.epochs}, Batchs {batch_idx + 1}/{step_num}, "
                    f"Training Loss: {loss.item() * args.gradient_accumulation_steps:9.4f}, "
                    f"Training Acc {batch_acc:9.4f}"
                )

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.warning("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.error(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    logger.info('saving model for epoch {}'.format(epoch + 1))
    model_path = join(args.save_model_path, 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_path)
    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, epoch, args):
    logger.info("start validating")
    model.eval()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}".format(epoch + 1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.warning("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.error(str(exception))
            raise exception


def train(tokenizer, model, train_dataset, validate_dataset, args):
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_model_path)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    num_warmup_steps = int(t_total * args.warmup_steps_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )
    logger.info('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    train_iterator = trange(int(args.epochs), desc='Epoch', mininterval=0)
    for epoch in train_iterator:
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, args=args)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader,
            epoch=epoch, args=args)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch + 1))
            model_path = join(args.save_model_path, 'min_ppl_model'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

        #  如果patience=0,则不进行early stopping
        if args.patience == 0:
            continue
        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))


def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--model_config', default='config/config.json', type=str, help='设置模型参数')
    parser.add_argument('--train_path', default='data/train.pkl', type=str, help='训练集路径')
    parser.add_argument('--max_len', default=150, type=int, help='训练时，输入数据的最大长度')
    parser.add_argument('--log_path', default='data/train.log', type=str, help='训练日志存放位置')
    parser.add_argument('--log', default=True, help="是否记录日志")
    parser.add_argument('--ignore_index', default=-100, type=int, help='对于ignore_index的label token不计算梯度')
    parser.add_argument('--epochs', default=100, type=int, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=16, type=int, help='训练的batch size')
    parser.add_argument('--gpu0_bsz', default=10, type=int, help='0号卡的batch size')
    parser.add_argument('--lr', default=2.6e-5, type=float, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, help='衰减率')
    parser.add_argument('--log_step', default=10, type=int, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=2.0, type=float)
    parser.add_argument('--save_model_path', default='./outputs/', type=str, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='uer/gpt2-distil-chinese-cluecorpussmall', type=str,
                        help='预训练的模型的路径')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping")
    parser.add_argument('--warmup_steps_rate', type=float, default=0.05, help='warm up步数')
    parser.add_argument('--val_rate', type=float, default=0.1, help='验证集大小')
    args = parser.parse_args()
    return args


def main():
    # 初始化参数
    args = set_args()
    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.cuda = not args.no_cuda

    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:0' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model)
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id

    # 创建模型的输出目录
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 创建模型
    if args.pretrained_model:  # 加载预训练模型
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:  # 初始化模型
        model_config = GPT2Config.from_json_file(args.model_config)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(device)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # 并行训练模型
    if args.cuda and torch.cuda.device_count() > 1:
        model = DataParallel(model).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset, validate_dataset = load_dataset(args)
    train(tokenizer, model, train_dataset, validate_dataset, args)


if __name__ == '__main__':
    main()
