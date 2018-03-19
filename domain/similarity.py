# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from domain.pos import POS_WEIGHT


def word_pos_similarity(l1, l2, similarity_type='word', pos_weight=None, embedding=None):
    """
    get similarity score by text vector and pos vector
    :param l1:
    :param l2:
    :param similarity_type:
    :param pos_weight:
    :param embedding:
    :return:
    """
    if not l1 or not l2:
        return 0
    pos_weight = pos_weight or POS_WEIGHT
    if similarity_type == 'word':
        # simple word name overlapping coefficient
        return len(set(l1) & set(l2)) / len(set(l1))
    elif similarity_type == 'word_pos':
        # word and pos overlapping coefficient
        sim_weight = 0
        for word, pos in set(l1):
            sim_weight += pos_weight.get(pos, 1) if word in l2 else 0
        total_weight = sum(pos_weight.get(pos, 1) for _, pos in set(l1))
        return sim_weight / total_weight if total_weight > 0 else 0
    elif similarity_type == 'vector' and embedding:
        # word vector and pos weight
        sim_weight = 0
        total_weight = 0
        for word, pos in l1:
            if word not in embedding.index2word:
                continue
            cur_weight = pos_weight.get(pos, 1)
            max_word_sim = max(embedding.similarity(word_l2, word) for word_l2 in l2)
            sim_weight += cur_weight * max_word_sim
            total_weight += cur_weight
        return sim_weight / total_weight if total_weight > 0 else 0
    return 0


if __name__ == '__main__':
    print(word_pos_similarity([1, 2], [3, 2]))
