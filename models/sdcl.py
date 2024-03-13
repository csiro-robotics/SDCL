import copy
import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Subspace distillation for continual learning.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='KD weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='SD weight.')
    return parser


class SDCL(ContinualModel):
    NAME = 'sdcl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(SDCL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.current_task = 0

    def compute_subspace_distance(self, feature1_, feature2_, labels1=None, labels2=None, dim=10):
        assert feature1_.shape == feature2_.shape
        batch_size = feature1_.shape[0]
        feature1_ = feature1_ + torch.zeros_like(feature1_).normal_(0, 0.01)
        feature2_ = feature2_ + torch.zeros_like(feature2_).normal_(0, 0.01)

        if labels2 is not None:
            labels_unique = torch.unique(labels2)
            label_ind1 = [[i for i, value in enumerate(labels1) if value == x] for x in labels_unique]
            label_ind2 = [[i for i, value in enumerate(labels2) if value == x] for x in labels_unique]

            dist = []

            for label_ind in range(len(label_ind2)):

                feature1 = feature1_[label_ind1[label_ind]]
                feature2 = feature2_[label_ind2[label_ind]]
		
                samples1 = torch.transpose(feature1, 0, 1)
                samples2 = torch.transpose(feature2, 0, 1)
		
                u_x, s_x, v_x = torch.linalg.svd(samples1, full_matrices=False)
                u_y, s_y, v_y = torch.linalg.svd(samples2, full_matrices=False)
		
                dist_per_class = 2 * dim - 2 * torch.norm(torch.mm(torch.transpose(u_x[:, 0:dim], 0, 1), u_y[:, 0:dim])) ** 2
                ## dist_per_class = (1/feature1.shape[0])*(2 * dim - 2 * torch.frobenius_norm(torch.mm(torch.transpose(u_x[:, 0:dim], 0, 1), u_y[:, 0:dim])) ** 2)
                dist.append(dist_per_class)

            dist = torch.tensor(dist).mean()

            return dist

        feature1 = torch.transpose(feature1_, 0, 1)
        feature2 = torch.transpose(feature2_, 0, 1)
        u_x, s_x, v_x = torch.linalg.svd(feature1, full_matrices=False)
        u_y, s_y, v_y = torch.linalg.svd(feature2, full_matrices=False)

        dist = 2 * dim - 2 * torch.norm(torch.mm(torch.transpose(u_x[:, 0:dim], 0, 1), u_y[:, 0:dim])) ** 2
        # dist = 2 * dim - 2 * torch.norm(torch.mm(torch.transpose(u_x, 0, 1), u_y)) ** 2

        return dist


    def begin_task(self, dataset):
        self.current_task += 1


    def end_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)
        

    def observe(self, inputs, labels, not_aug_inputs, use_sd=True, cwise_sd=True):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs_, buf_features_ = self.net(buf_inputs, returnt='all')

            # loss += self.args.alpha * F.mse_loss(buf_outputs_, buf_logits)
            loss += self.args.alpha * self.loss(buf_outputs_, buf_labels)

        if not self.buffer.is_empty() and self.old_net is not None and use_sd:
            if self.current_task > 1:
                buf_output, buf_features = self.old_net(buf_inputs, returnt='all')

            if cwise_sd: 
                loss +=  self.args.beta * self.compute_subspace_distance(buf_features_, buf_features.detach(), buf_labels, buf_labels)
            else: 
                loss +=  self.args.beta * self.compute_subspace_distance(buf_features_, buf_features.detach())

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)

        return loss.item()
