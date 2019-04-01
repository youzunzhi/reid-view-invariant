from model import resnet50, Resnet50WithDomainClassifier
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
    def __init__(self, args, source_num_classes, logger):
        super(ReverseLayerTrainer, self).__init__()
        self.args = args
        self.source_num_classes = source_num_classes
        self.logger = logger

        self.net = Resnet50WithDomainClassifier(self.source_num_classes).cuda() if self.args.use_cuda else Resnet50WithDomainClassifier(self.source_num_classes)
        self.label_loss = nn.CrossEntropyLoss().cuda() if self.args.use_cuda else nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss().cuda() if self.args.use_cuda else nn.CrossEntropyLoss()

        self.logger.print_log('do not use pre-trained model. train from scratch.')

        bn_params, other_params = partition_params(self.net, 'bn')
        self.net_optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                      {'params': other_params}], lr=args.lr, momentum=0.9, weight_decay=args.wd)
        self.recorder = SummaryWriter(os.path.join(args.save_path, 'tb_logs'))

    def train_epoch(self, source_train_loader, target_train_loader, epoch, total_epoch):
        adjust_learning_rate(self.net_optimizer, (self.args.lr,), epoch, self.args.epochs, self.args.lr_strategy)
        batch_time_meter = AverageMeter()
        stats = ('acc/r1', 'loss', 'source_label_loss', 'source_domain_loss', 'target_domain_loss')
        meters_trn = {stat: AverageMeter() for stat in stats}
        self.train()
        end = time.time()

        len_train_loader = min(len(source_train_loader), len(target_train_loader))
        source_train_iter = iter(source_train_loader)
        target_train_iter = iter(target_train_loader)

        for i in range(len_train_loader):
            p = float(i + epoch * len_train_loader) / total_epoch / len_train_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            source_tuple = source_train_iter.next()
            source_img = source_tuple[0].cuda() if self.args.use_cuda else source_tuple[0]
            source_labels = source_tuple[1].cuda() if self.args.use_cuda else source_tuple[1]

            batch_size = len(source_labels)
            source_domain = torch.zeros(batch_size, dtype=torch.long).cuda() if self.args.use_cuda else torch.zeros(batch_size, dtype=torch.long)

            _, labels_pred, domains_pred = self.net(source_img, alpha)
            source_label_loss = self.label_loss(labels_pred, source_labels)
            source_domain_loss = self.domain_loss(domains_pred, source_domain)

            # training model using target data
            target_tuple = target_train_iter.next()
            target_img = target_tuple[0].cuda() if self.args.use_cuda else target_tuple[0]

            batch_size = len(target_img)
            target_domain = torch.ones(batch_size, dtype=torch.long).cuda() if self.args.use_cuda else torch.ones(batch_size, dtype=torch.long)

            _, _, domains_pred = self.net(target_img, alpha)
            target_domain_loss = self.domain_loss(domains_pred, target_domain)
            loss = source_label_loss + source_domain_loss + target_domain_loss

            self.net_optimizer.zero_grad()
            loss.backward()
            self.net_optimizer.step()

            acc = compute_accuracy(labels_pred, source_labels)
            meters_trn['acc/r1'].update(acc, self.args.batch_size)
            meters_trn['loss'].update(loss.item(), self.args.batch_size)
            meters_trn['source_label_loss'].update(source_label_loss.item(), self.args.batch_size)
            meters_trn['source_domain_loss'].update(source_domain_loss.item(), self.args.batch_size)
            meters_trn['target_domain_loss'].update(target_domain_loss.item(), self.args.batch_size)
            batch_time_meter.update(time.time() - end)
            freq = self.args.batch_size / batch_time_meter.avg
            end = time.time()
            if i % self.args.print_freq == 0:
                self.logger.print_log('  Iter: [{:03d}/{:03d}]   Freq {:.1f}   '.format(
                    i, len_train_loader, freq) + create_stat_string(meters_trn) + time_string())
        if len(source_train_loader) < len(target_train_loader):
            len_target_rest = len(target_train_loader)-len_train_loader
            for i in range(len_target_rest):
                target_tuple = target_train_iter.next()
                target_img = target_tuple[0].cuda() if self.args.use_cuda else target_tuple[0]
                batch_size = len(target_img)
                target_domain = torch.ones(batch_size, dtype=torch.long).cuda() if self.args.use_cuda else torch.ones(
                    batch_size, dtype=torch.long)

                _, _, domains_pred = self.net(target_img, alpha)
                target_domain_loss = self.domain_loss(domains_pred, target_domain)
                loss = target_domain_loss

                self.net_optimizer.zero_grad()
                loss.backward()
                self.net_optimizer.step()
                meters_trn['target_domain_loss'].update(target_domain_loss.item(), self.args.batch_size)
                if i % self.args.print_freq == 0 and i != 0 :
                    self.logger.print_log('  Iter: [{:03d}/{:03d}] target_loss:{} '.format(
                        i, len_target_rest, meters_trn['target_domain_loss'].avg) + time_string())

        self.recorder.add_scalars("stats/acc_r1", {'train_acc': meters_trn['acc/r1'].avg}, epoch)
        self.recorder.add_scalar("loss/loss_cls", meters_trn['loss'].avg, epoch)
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



