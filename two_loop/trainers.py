from model import resnet50, Resnet50WithViewClassifier
from utils import *
import torch.nn as nn
import torch
import os
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self, *names):
        """
        set the given attributes in names to the training state.
        if names is empty, call the train() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).train()

    def eval(self, *names):
        """
        set the given attributes in names to the evaluation state.
        if names is empty, call the eval() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).eval()


class ReidTrainer(Trainer):
    def __init__(self, args, num_classes, logger):
        super(ReidTrainer, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.logger = logger

        self.net = resnet50(pretrained=False, num_classes=num_classes).cuda()
        if args.pretrain_path is None:
            self.logger.print_log('do not use pre-trained model. train from scratch.')
        elif os.path.isfile(args.pretrain_path):
            checkpoint = torch.load(args.pretrain_path)
            fixed_layers = ('fc',)
            state_dict = reset_state_dict(checkpoint, self.net, *fixed_layers)
            self.net.load_state_dict(state_dict)
            self.logger.print_log('loaded pre-trained model from {}'.format(args.pretrain_path))
        else:
            self.logger.print_log('{} is not a file. train from scratch.'.format(args.pretrain_path))

        self.cls_loss = nn.CrossEntropyLoss().cuda()

        bn_params, other_params = partition_params(self.net, 'bn')
        self.optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                         {'params': other_params}], lr=args.lr, momentum=0.9, weight_decay=args.wd)

        self.recorder = SummaryWriter(os.path.join(args.save_path, 'tb_logs'))

    def train_epoch(self, train_loader, epoch):

        adjust_learning_rate(self.optimizer, (self.args.lr,), epoch, self.args.epochs, self.args.lr_strategy)
        batch_time_meter = AverageMeter()
        stats = ('acc/r1', 'loss_cls')
        meters_trn = {stat: AverageMeter() for stat in stats}
        self.train()
        end = time.time()

        for i, train_tuple in enumerate(train_loader):
            imgs = train_tuple[0].cuda()
            labels = train_tuple[1].cuda()

            predictions = self.net(imgs)[1]
            loss = self.cls_loss(predictions, labels)
            acc = compute_accuracy(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i == 0 and self.args.record_grad:
                name = 'conv1.weight'
                param = self.net.conv1.weight
                self.recorder.add_histogram(name+'_grad', param.grad, epoch, bins='auto')

            stats = {'acc/r1': acc,
                     'loss_cls': loss.item()}
            for k, v in stats.items():
                meters_trn[k].update(v, self.args.batch_size)

            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len(train_loader), freq) + create_stat_string(meters_trn) + time_string())

        self.recorder.add_scalars("stats/acc_r1", {'train_acc': meters_trn['acc/r1'].avg}, epoch)
        self.recorder.add_scalar("loss/loss_cls", meters_trn['loss_cls'].avg, epoch)

        save_checkpoint(self, epoch, os.path.join(self.args.save_path, "checkpoints.pth"))
        return meters_trn

    def eval_performance(self, gallery_loader, probe_loader, epoch):
        stats = ('acc/r1', 'mAP')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net, index_feature=0)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net, index_feature=0)
        dist = cdist(gallery_features, probe_features, metric='cosine')
        CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
        rank1 = CMC[0]
        meters_val['acc/r1'].update(rank1, 1)
        meters_val['mAP'].update(MAP, 1)
        self.recorder.add_scalars("stats/acc_r1", {'eval_r1': meters_val['acc/r1'].avg}, epoch)
        return meters_val


class ReverseLayerTrainer(Trainer):
    def __init__(self, args, num_classes, num_views, logger):
        super(ReverseLayerTrainer, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.num_views = num_views
        self.logger = logger

        self.net = Resnet50WithViewClassifier(num_classes, num_views).cuda() if self.args.use_cuda else Resnet50WithViewClassifier(num_classes, num_views)
        self.view_loss = nn.CrossEntropyLoss().cuda() if self.args.use_cuda else nn.CrossEntropyLoss()
        self.label_loss = nn.CrossEntropyLoss().cuda() if self.args.use_cuda else nn.CrossEntropyLoss()

        self.logger.print_log('do not use pre-trained model. train from scratch.')

        bn_params, other_params = partition_params(self.net, 'bn')
        self.net_optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                      {'params': other_params}], lr=args.lr, momentum=0.9, weight_decay=args.wd)
        self.recorder = SummaryWriter(os.path.join(args.save_path, 'tb_logs'))

    def train_epoch(self, train_view_loader, train_label_loader, epoch, total_epoch):
        adjust_learning_rate(self.net_optimizer, (self.args.lr,), epoch, self.args.epochs, self.args.lr_strategy)
        batch_time_meter = AverageMeter()
        stats = ('acc/r1', 'label_loss', 'view_loss')
        meters_trn = {stat: AverageMeter() for stat in stats}
        self.train()
        end = time.time()

        for i, train_tuple in enumerate(train_view_loader):
            p = float(i + epoch * len(train_view_loader)) / total_epoch / len(train_view_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            imgs = train_tuple[0].cuda() if self.args.use_cuda else train_tuple[0]
            views = train_tuple[1].cuda() if self.args.use_cuda else train_tuple[2]
            views = views - 1 # view1-15 to view0-14
            _, _, views_pred = self.net(imgs, alpha)
            view_loss = self.view_loss(views_pred, views)
            loss = view_loss
            self.net_optimizer.zero_grad()
            loss.backward()
            self.net_optimizer.step()

            meters_trn['view_loss'].update(view_loss.item(), self.args.batch_size)
            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len(train_view_loader), freq) + create_stat_string(meters_trn) + time_string())

        for i, train_tuple in enumerate(train_label_loader):
            p = float(i + epoch * len(train_label_loader)) / total_epoch / len(train_label_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            imgs = train_tuple[0].cuda() if self.args.use_cuda else train_tuple[0]
            labels = train_tuple[1].cuda() if self.args.use_cuda else train_tuple[1]
            _, labels_pred, _ = self.net(imgs, alpha)
            label_loss = self.label_loss(labels_pred, labels)
            loss = label_loss
            acc = compute_accuracy(labels_pred, labels)
            self.net_optimizer.zero_grad()
            loss.backward()
            self.net_optimizer.step()

            meters_trn['acc/r1'].update(acc, self.args.batch_size)
            meters_trn['label_loss'].update(label_loss.item(), self.args.batch_size)
            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len(train_label_loader), freq) + create_stat_string(meters_trn) + time_string())

        self.recorder.add_scalars("stats/acc_r1", {'train_acc': meters_trn['acc/r1'].avg}, epoch)
        self.recorder.add_scalar("loss/loss_cls", meters_trn['label_loss'].avg, epoch)
        save_checkpoint(self, epoch, os.path.join(self.args.save_path, "checkpoints.pth"))
        return meters_trn

    def eval_performance(self, gallery_loader, probe_loader, epoch):
        stats = ('acc/r1', 'mAP')
        meters_val = {stat: AverageMeter() for stat in stats}
        self.eval()
        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net, index_feature=0)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net, index_feature=0)
        dist = cdist(gallery_features, probe_features, metric='cosine')
        CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views)
        rank1 = CMC[0]
        meters_val['acc/r1'].update(rank1, 1)
        meters_val['mAP'].update(MAP, 1)
        self.recorder.add_scalars("stats/acc_r1", {'eval_r1': meters_val['acc/r1'].avg}, epoch)
        return meters_val


