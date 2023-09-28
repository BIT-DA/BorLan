# simplified version
import sys
import os
proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.classifier import Classifier
from models.method import BorLan
from tensorboardX import SummaryWriter
from src.utils import load_network, load_data
from src.classnames import *

from clip import clip
from src.imagenet_templates import IMAGENET_TEMPLATES

def pseudo_labeling(loader, model, classifier, device):
    with torch.no_grad():
        start_test = True
        iter_val = iter(loader['unlabeled_test'])
        for i in range(len(loader['unlabeled_test'])):
            data = iter_val.next()
            inputs = data[0][0]
            labels = data[1]
            indexs = data[2]
            inputs = inputs.to(device)
            labels = labels.to(device)
            feat = model.inference(inputs)
            outputs = classifier(feat)

            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                all_index = indexs.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
                all_index = torch.cat((all_index, indexs.data.float()), 0)

        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        # print(predict, all_label)
    return accuracy, predict

def test_cifar(loader, model, classifier, device):
    with torch.no_grad():
        start_test = True
        iter_val = iter(loader['test'])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            feat = model.inference(inputs)
            outputs = classifier(feat)

            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def test(loader, model, classifier, device):
    with torch.no_grad():
        model.eval()
        classifier.eval()
        start_test = True
        val_len = len(loader['test0'])
        iter_val = [iter(loader['test' + str(i)]) for i in range(10)]
        for _ in range(val_len):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)
            outputs = []
            for j in range(10):
                feat = model.inference(inputs[j])
                output = classifier(feat)
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_outputs = outputs.data.float()
                all_labels = labels.data.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.data.float()), 0)
                all_labels = torch.cat((all_labels, labels.data.float()), 0)
        _, predict = torch.max(all_outputs, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])
    return accuracy

def train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=None, writer=None, model_path = None):

    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])

    len_unlabeled = len(dataset_loaders["unlabeled_train"])
    iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    best_acc = 0.0
    best_model = None

    pseudo_labels = None
    for iter_num in range(1, args.max_iter + 1):
        if (iter_num-1) % len_unlabeled == 0:
            model.eval()
            classifier.eval()
            pacc, pseudo_labels = pseudo_labeling(dataset_loaders, model, classifier, device=device)
            print('iter: {}, pseudo labeling acc: {:.2f}'.format(iter_num, pacc*100))
    
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
        if iter_num % len_unlabeled == 0:
            iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

        data_labeled = iter_labeled.next()
        data_unlabeled = iter_unlabeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)

        img_unlabeled_q = data_unlabeled[0][0].to(device)
        img_unlabeled_k = data_unlabeled[0][1].to(device)
        index = data_unlabeled[2]
        plabel = pseudo_labels[index].to(device)

        ## For Labeled Data
        img_labeled = torch.cat([img_labeled_q,img_labeled_k],dim=0)
        emb_labeled, feat_labeled = model(img_labeled)
        emb_labeled_q, emb_labeled_k = emb_labeled.chunk(2,dim=0)
        feat_labeled_q, feat_labeled_k = feat_labeled.chunk(2,dim=0)
        out_q = classifier(feat_labeled_q)
        out_k = classifier(feat_labeled_k)
        classifier_loss = criterions['CrossEntropy'](out_q, label)
        classifier_loss += criterions['CrossEntropy'](out_k, label)
 
        align_loss_labeled = InfiniteContrastiveV2(args, emb_labeled_q, label)
        align_loss_labeled += InfiniteContrastiveV2(args, emb_labeled_k, label)

        ## For Unlabeled Data
        predict_unlabeled = plabel
        
        img_unlabeled = torch.cat([img_unlabeled_q,img_unlabeled_k],dim=0)
        emb_unlabeled, feat_unlabeled = model(img_unlabeled)
        emb_unlabeled_q, emb_unlabeled_k = emb_unlabeled.chunk(2,dim=0)
        align_loss_unlabeled = InfiniteContrastiveV2(args, emb_unlabeled_q, predict_unlabeled)
        align_loss_unlabeled += InfiniteContrastiveV2(args, emb_unlabeled_k, predict_unlabeled)

        total_loss = classifier_loss*args.cx + align_loss_labeled*args.cl + align_loss_unlabeled*args.cu
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out_q, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))

        ## Show Loss in TensorBoard
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/total_loss', total_loss, iter_num)

        if iter_num % args.test_interval == 1:# or iter_num == 500:
            if iter_num == 1:
                continue
            model.eval()
            classifier.eval()
            if 'cifar100' in args.root:
                test_acc = test_cifar(dataset_loaders, model, classifier, device=device)
            else:
                test_acc = test(dataset_loaders, model, classifier, device=device)
            print("iter_num: {}; test_acc: {}".format(iter_num, test_acc))
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = {'model': model.state_dict(),
                              'classifier': classifier.state_dict(),
                              'step': iter_num
                              }
    print("best acc: %.4f" % (best_acc))
    torch.save(best_model, model_path)
    print("The best model has been saved in ", model_path)

def InfiniteContrastive(args,query,label):
    T = 0.07
    label_onehot = F.one_hot(label, args.class_num).float().cuda()
    # print(self.means)
    # print(self.covs)
    key_means = label_onehot @ args.means.float()

    key_covs = []
    for i, cls in enumerate(label):
        key_covs.append(args.covs[cls])
    key_covs = torch.stack(tuple(key_covs), dim=0)
    # print("key cov", key_covs.size())

    l_pos = torch.einsum('nc,nc->n',
                            [query, key_means + 0.5 * torch.bmm((key_covs/T), query.unsqueeze(dim=-1)).squeeze(dim=-1)]).unsqueeze(-1)
    # negative logits: NxK
    l_neg = [torch.einsum('nc,ck->nk', [query, args.means.T]), -torch.bmm(query.unsqueeze(dim=1), key_means.unsqueeze(dim=-1)).squeeze(dim=-1)]
    l_neg = torch.cat(l_neg,dim=1)

    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= T

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    # dequeue and enqueue
    N, C = query.shape
    jcl_loss = 0.5 * torch.bmm(torch.bmm(query.reshape(N, 1, C), (key_covs/T)), query.reshape(N, C, 1)).mean() / T
    loss = nn.CrossEntropyLoss()(logits,labels) + jcl_loss
    return loss

def InfiniteContrastiveV2(args, query, label):
    T = 0.07
    query_mean = query.mm(args.means.permute(1,0).float()) #N*K
    covs = args.covs / T
    query_cov_query = 0.5*query.pow(2).mm(covs.permute(1,0))
    logits = query_mean + query_cov_query

    # apply temperature
    logits /= T
    ce_loss = F.cross_entropy(logits, label, reduction='none')

    key_covs = covs[label]
    jcl_loss = (0.5 * torch.sum(query.pow(2).mul(key_covs), dim=1)) / T
    # return (F.cross_entropy(logits, labels, reduction='none')*mask).mean() + jcl_loss
    loss = ce_loss + jcl_loss
    return loss.mean()

# obtain class-wise mean and covariance from saved features
# all_text_features.size() = Class, Nprompts, dim
def text_embedding(filename):
    all_text_features = torch.load(filename)

    all_means = []
    all_covs = []
    for c in range(all_text_features.size(0)):
        cls_text_features = all_text_features[c].cpu().numpy()
        mean = np.mean(cls_text_features, axis=0) # 1024 
        cov = np.cov(cls_text_features.T) # 1024 x 1024
        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(cov)
        cov = cov.diag()
        all_means.append(mean)
        all_covs.append(cov)
    means = torch.stack(tuple(all_means), dim=0).float().cuda()
    covs = torch.stack(tuple(all_covs), dim=0).float().cuda()
    return means, covs

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--label_ratio', type=int, default=15)
    parser.add_argument('--logdir', type=str, default='vis/')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--cx', type=float, default='1.0')
    parser.add_argument('--cl', type=float, default='1.0')
    parser.add_argument('--cu', type=float, default='1.0')
    parser.add_argument('--seed', type=int, default='666666')
    parser.add_argument('--workers', type=int, default='4')
    parser.add_argument('--lr_ratio', type=float, default='10')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--projector_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=27005)
    parser.add_argument('--test_interval', type=float, default=3000)
    parser.add_argument("--pretrained", action="store_true", help="use the pre-trained model")
    parser.add_argument("--pretrained_path", type=str, default='.')
    parser.add_argument('--num_labeled', type=int, default=0, help='number of labeled data')
    parser.add_argument("--text_model_name", type=str, default='bert')
    ## Only for Cifar100
    parser.add_argument("--expand_label", action="store_true", help="expand label to fit eval steps")
    
    configs = parser.parse_args()
    return configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    args = read_config()
    set_seed(args.seed)

    #means, covs = build_clip_model(class_names=CLASSES_home)
    #return
    # Prepare data
    if 'cifar100' in args.root:
        args.dataset = 'cifar100'
        args.class_num = 100
        args.classname = CLASSES_Cifar100
    elif 'CUB200' in args.root:
        args.dataset = 'bird'
        args.class_num = 200
        args.classname = CLASSES_Birds
    elif 'StanfordCars' in args.root:
        args.dataset = 'cars'
        args.class_num = 196
        args.classname = CLASSES_Car
    elif 'Aircraft' in args.root:
        args.dataset = 'aircraft'
        args.class_num = 100
        args.classname = CLASSES_AirCraft
    args.data_return_index = True

    dataset_loaders = load_data(args)
    print("class_num: ", args.class_num)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    if 'cifar100' in args.root:
        model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.num_labeled))
    else:
        model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.label_ratio))
    logdir = os.path.join(args.logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    model_path = os.path.join(logdir, "%s_best.pkl" % (model_name))

    # Initialize model
    network, feature_dim = load_network(args.backbone)
    model = BorLan(network=network, backbone=args.backbone, projector_dim=args.projector_dim,        feature_dim=feature_dim, class_num=args.class_num, pretrained=args.pretrained, pretrained_path=args.pretrained_path).to(device)
    classifier = Classifier(feature_dim, args.class_num).to(device)
    print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))

    ## Define Optimizer
    base_params = []
    proj_params = []
    for pname, p in model.named_parameters():
        if pname.startswith('self.encoder.proj'):
            proj_params += [p]
        else:
            base_params += [p]
    optimizer = optim.SGD([
        #{'params': model.parameters()},
        {'params': base_params},
        {'params': proj_params, 'lr': args.lr * args.lr_ratio},
        {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr= args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    names = {'clip':'_clipL.pt','bert':'_BertL.pt', 'mT5':'_mT5L.pt'}
    filename = args.dataset + names[args.text_model_name]
    print('using text model: ',filename)
    means, covs = text_embedding(filename)
    args.means = means
    args.covs = covs

    # Train model
    train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=device, writer=writer, model_path=model_path)

if __name__ == '__main__':
    main()
