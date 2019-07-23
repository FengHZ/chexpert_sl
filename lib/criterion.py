import torch
from torch import nn
import torch.nn.functional as F


# eps = 1e-7


class ClsCriterion(nn.Module):
    def __init__(self, class_weight=torch.ones(5)):
        """
        :param class_weight: the weight for each class
        """
        super(ClsCriterion, self).__init__()
        self.class_weight = class_weight.cuda()

    def forward(self, classify_result_list, label_weight, label):
        """
        :param classify_result_list: the classification result for each class(log softmax mode)
        [predict1,predict2,...,predictn]
        :param label_weight:N*n
        :param label:N*n not one hot
        :return:
        """
        loss = torch.zeros(self.class_weight.size())
        if label.is_cuda:
            loss = loss.cuda()
        for idx, item in enumerate(classify_result_list):
            mask_label = torch.zeros(item.size())
            if label.is_cuda:
                mask_label = mask_label.cuda()
            mask_label.scatter_(1, label[:, idx].view(-1, 1), 1)
            cls_loss = -1 * torch.mean(torch.sum(item * mask_label, dim=1) * label_weight[:, idx])
            loss[idx] = cls_loss
        loss = torch.mean(loss * self.class_weight)
        return loss




