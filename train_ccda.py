import os
import os.path as osp
import time
import random
import warnings
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from loss import *
import lr_schedule
from logger import Logger
import pre_process as prep
from network import resnet50, resnet101
from tools import image_classification_test
from datasets import CB_data_batch, ImageList, SBSList

cudnn.benchmark = True
# cudnn.deterministic = True
warnings.filterwarnings('ignore')

def train(config):
    # pre-process training and test data
    prep_dict = {'source': prep.image_train(**config['prep']['params']),
                 'target': prep.image_train(**config['prep']['params'])}
    if config['prep']['test_10crop']:
        prep_dict['test'] = prep.image_test_10crop(**config['prep']['params'])
    else:
        prep_dict['test'] = prep.image_test(**config['prep']['params'])

    data_set = {}
    dset_loaders = {}
    data_config = config['data']  
    
    data_set['source'] = ImageList(open(data_config['source']['list_path']).readlines(),
                                    transform=prep_dict['source'], is_train=True, max_iter=config['max_iter'])  
    data_set['target'] = ImageList(open(data_config['target']['list_path']).readlines(),
                                   transform=prep_dict['target'], is_train=True, max_iter=config['max_iter'],batchsize=data_config['source']['batch_size'])
        
    # random batch sampler or class-balance sampler for source dataloader
    if config['data']['sampler'] == 'random':
        dataloader_dict = {'batch_size': data_config['source']['batch_size'],
                           'shuffle': True, 'drop_last': True, 'num_workers': 16}
    elif config['data']['sampler'] == 'sbs':
        data_set['source'] = SBSList(open(data_config['source']['list_path']).readlines(),
                                   transform=prep_dict['source'],
                                   max_iter=config['max_iter'],
                                   batchsize=data_config['source']['batch_size'],
                                   log=config['logger'])
        dataloader_dict = {'batch_size': data_config['source']['batch_size'],
                           'shuffle': True, 'drop_last': True, 'num_workers': 16}          
    elif config['data']['sampler'] == 'sbs_fixed':
        data_set['source'] = SBSList(open(data_config['source']['list_path']).readlines(),
                                   transform=prep_dict['source'],
                                   max_iter=config['max_iter'],
                                   batchsize=data_config['source']['batch_size'],
                                   log=config['logger'],
                                   fixed_temp=config['temp'])
        dataloader_dict = {'batch_size': data_config['source']['batch_size'],
                           'shuffle': True, 'drop_last': True, 'num_workers': 16} 
    elif config['data']['sampler'] == 'cls_balance':
        s_gt = np.array(data_set['source'].imgs)[:, 1]
        b_sampler = CB_data_batch(s_gt, batch_size=data_config['source']['batch_size'], 
                                drop_last=False, gt_flag=True, 
                                num_class=config['network']['params']['class_num'], 
                                num_batch=data_config['source']['batch_size'])
        dataloader_dict = {'batch_sampler' : b_sampler, 'num_workers': 16}
    else:
        raise ValueError('sampler %s not found!' % (config['data']['sampler'])) 
    
    dset_loaders['source'] = torch.utils.data.DataLoader(data_set['source'], 
                                                         **dataloader_dict)
    dset_loaders['target'] = torch.utils.data.DataLoader(data_set['target'],
                                                         batch_size=data_config['target']['batch_size'],
                                                         shuffle=True, num_workers=16, drop_last=True)
    
    if config['prep']['test_10crop']:
        data_set['test'] = [ImageList(open(data_config['test']['list_path']).readlines(),
                                      transform=prep_dict['test'][i]) for i in range(10)]
        dset_loaders['test'] = [torch.utils.data.DataLoader(dset, batch_size=data_config['test']['batch_size'],
                                                            shuffle=False, num_workers=16) for dset in data_set['test']]
    else:
        data_set['test'] = ImageList(open(data_config['test']['list_path']).readlines(), transform=prep_dict['test'])
        dset_loaders['test'] = torch.utils.data.DataLoader(data_set['test'],
                                                           batch_size=data_config['test']['batch_size'],
                                                           shuffle=False, num_workers=16)

    # set base network, classifier network, residual net
    class_num = config['network']['params']['class_num']
    net_config = config['network']
    if net_config['name'] == '50':
        base_network = resnet50()
    elif net_config['name'] == '101':
        base_network = resnet101()
    else:
        raise ValueError('base network %s not found!' % (net_config['name']))
    base_network = base_network.cuda()
    classifier_layer = nn.Linear(base_network.out_features, class_num)
    classifier_layer = classifier_layer.cuda()
    classifier_layer.weight.data.normal_(0, 0.01)
    classifier_layer.bias.data.fill_(0.0)
    softmax_layer = nn.Softmax().cuda()
    
    # set optimizer
    parameter_list = [
        {'params': base_network.parameters(), 'lr_mult': 1, 'decay_mult': 2},
        {'params': classifier_layer.parameters(), 'lr_mult': 10, 'decay_mult': 2}
    ]
    optimizer_config = config['optimizer']
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    schedule_param = optimizer_config['lr_param']
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config['lr_type']]

    # set loss
    class_criterion = nn.CrossEntropyLoss().cuda()
    loss_config = config['loss']
    if 'params' not in loss_config:
        loss_config['params'] = {}

    # train
    iter_source = iter(dset_loaders['source'])
    iter_target = iter(dset_loaders['target'])
    best_acc = 0.0
    since = time.time()
    for num_iter in tqdm(range(config['max_iter'])):
        if num_iter % config['val_iter'] == 0 and num_iter != 0:
            base_network.train(False)
            classifier_layer.train(False)
            temp_acc, s_acc, s_recall, p_acc, p_recall = image_classification_test(loader=dset_loaders, base_net=base_network,
                                                         classifier_layer=classifier_layer,
                                                         test_10crop=config['prep']['test_10crop'],
                                                         config=config, num_iter=num_iter
                                                         )
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = {'base': base_network.state_dict(), 'classifier': classifier_layer.state_dict()}
            log_str = 'iter: {:d}, all_accu: {:.4f},\ttime: {:.4f}'.format(num_iter, temp_acc, time.time() - since)
            config['logger'].logger.debug(log_str)
            config['results'][num_iter].append(temp_acc)
            log_str = 'shared_acc: {:.4f},\tshared_recall: {:.4f},\tprivate_acc: {:.4f},\tprivate_recall: {:.4f}'.format(s_acc, s_recall, p_acc, p_recall)
            config['logger'].logger.debug(log_str)
            
        # This has any effect only on modules such as Dropout or BatchNorm.
        base_network.train(True)
        classifier_layer.train(True)

        # freeze BN layers
        for m in base_network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.training = False
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        # load data
        inputs_source, labels_source = next(iter_source)
        inputs_target, _ = next(iter_target)
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        # index for shared and privated class
        shared_index = labels_source < config['network']['params']['shared_class_num']
        privated_index = ~shared_index

        # update lr and optimizer
        optimizer = lr_scheduler(optimizer, num_iter / config['max_iter'], **schedule_param)
        optimizer.zero_grad()

        # network forward
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features_base = base_network(inputs)
        batch_size = data_config['test']['batch_size']
        
        # source classification loss
        output_base = classifier_layer(features_base)
        classifier_loss = class_criterion(output_base[:batch_size, :], labels_source)

        # target entropy loss
        softmax_output_target = softmax_layer(output_base[batch_size:, :])
        entropy_loss = EntropyLoss(softmax_output_target)

        # alignment of L task-specific feature layers (Here, we have one layer)
        Fatt2 = 0.0
        Frep = 0.0
        if sum(shared_index) > 0:
            Fatt1 = MMD(features_base[:batch_size, :][shared_index],
                            features_base[batch_size:, :])
            if sum(privated_index) > 0:
                Frep = MMD(features_base[:batch_size, :][shared_index],
                                features_base[:batch_size, :][privated_index])
                Fatt2 = MMD(features_base[:batch_size, :][privated_index], features_base[batch_size:, :])

                transfer_loss = config['loss']['alpha1_off'] * Fatt1 + \
                                config['loss']['alpha2_off'] * Fatt2  + \
                                config['loss']['alpha3_off'] * (-Frep)
            else:
                transfer_loss = config['loss']['alpha1_off'] * Fatt1
        else:
            transfer_loss = 0.0
            
        # total loss and update network
        total_loss = classifier_loss + \
                     transfer_loss + \
                     config['loss']['beta_off'] * entropy_loss
        total_loss.backward()
        optimizer.step()

        if num_iter % config['val_iter'] == 0:
            backbone_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
            config['logger'].logger.debug(
                'class: {:.4f}\tdis: {:.4f}\tentropy: {:.4f}\tFatt1: {:.4f}\tFrep: {:.4f}\tFatt2: {:.4f}'.format(classifier_loss.item(), 
                                                                     transfer_loss,
                                                                     config['loss']['beta_off'] * entropy_loss.item(),
                                                                     config['loss']['alpha1_off'] * Fatt1,
                                                                     config['loss']['alpha3_off'] * (-Frep),
                                                                     config['loss']['alpha2_off'] * Fatt2))
            config['logger'].logger.debug(
                'backbone_lr: {:.4f}\tclassifier_lr: {:.4f}'.format(backbone_lr, classifier_lr))

    torch.save(best_model, osp.join(config['path']['model'], config['task'] + '_best_model.pth'))
    return best_acc


def empty_dict(config):
    config['results'] = {}
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        config['results'][key] = []
    config['results']['best'] = []


def print_dict(config):
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        log_str = 'setting: {:d}, average: {:.4f}'.format(key, np.average(config['results'][key]))
        config['logger'].logger.debug(log_str)
    log_str = 'best, average: {:.4f}'.format(np.average(config['results']['best']))
    config['logger'].logger.debug(log_str)
    config['logger'].logger.debug('-' * 100)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='class-customized domain adaptation')
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument('--gpu', type=str, nargs='?', default='0', help='device id to run')
    parser.add_argument('--net', type=str, default='50', choices=['50', '101'])
    parser.add_argument('--sampler', type=str, default='sbs', choices=['random', 'sbs', 'cls_balance', 'sbs_fixed'])
    parser.add_argument('--data_set', default='office_10_10', choices=['office_10_10', 'home_10_10', 'domainnet_10_10', 'office_5_15', 'office_15_5', 'office_3shot', 'fruit_5_5', 'dog_5_5'], help='data set')
    parser.add_argument('--source_path', type=str, help='The source list')
    parser.add_argument('--target_path', type=str, help='The target list')
    parser.add_argument('--test_path', type=str, help='The test list')
    parser.add_argument('--output_path', type=str, default='snapshot/', help='save ``log/scalar/model`` file path')
    parser.add_argument('--task', type=str, default='da', help='transfer task name')
    parser.add_argument('--max_iter', type=int, default=20001, help='max iterations')
    parser.add_argument('--val_iter', type=int, default=500, help='interval of two continuous test phase')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (Default:1e-4')
    parser.add_argument('--batch_size', type=int, default=36, help='mini batch size')
    parser.add_argument('--beta_off', type=float, default=0.1, help='target entropy loss weight ')
    parser.add_argument('--alpha1_off', type=float, default=1.5, help='source to target align loss weight')
    parser.add_argument('--alpha2_off', type=float, default=1.5, help='target labeld to unlabled align loss weight')
    parser.add_argument('--alpha3_off', type=float, default=0.75, help='dis-alignment between source and target labeld data loss weight')
    parser.add_argument('--temp', type=float, default=1000.0, help='the value of temperature in SBS')
    args = parser.parse_args()

    os.environ['PYTHONASHSEED'] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # seed for everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    config = {'seed': args.seed, 'gpu': args.gpu, 'max_iter': args.max_iter, 'val_iter': args.val_iter,
              'data_set': args.data_set, 'task': args.task,
              'prep': {'test_10crop': True, 'params': {'resize_size': 256, 'crop_size': 224}},
              'network': {'name': args.net, 'params': {'resnet_name': args.net, 'class_num': 20, 'shared_class_num': 10}},
              'optimizer': {'type': optim.SGD,
                            'optim_params': {'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
                            'lr_type': 'inv', 'lr_param': {'lr': args.lr, 'gamma': 1.0, 'power': 0.75}},
              'data': {
                  'source': {'list_path': args.source_path, 'batch_size': args.batch_size},
                  'target': {'list_path': args.target_path, 'batch_size': args.batch_size},
                  'test': {'list_path': args.test_path, 'batch_size': args.batch_size},
                  'sampler': args.sampler},
              'output_path': args.output_path + args.data_set,
              'path': {'log': args.output_path + args.data_set + '/log/',
                       'model': args.output_path + args.data_set + '/model/'},
              'loss': {'alpha1_off': args.alpha1_off,
                       'alpha2_off': args.alpha2_off,
                       'alpha3_off': args.alpha3_off,
                       'beta_off': args.beta_off},
              'temp': args.temp,
              }
    
    if config['data_set'] == 'office_10_10' or config['data_set'] == 'home_10_10' or config['data_set'] == 'domainnet_10_10' or config['data_set'] == 'office_3shot':
        pass
    elif config['data_set'] == 'office_5_15':
        config['network']['params']['shared_class_num'] = 5   
    elif config['data_set'] == 'office_15_5':
        config['network']['params']['shared_class_num'] = 15            
    elif config['data_set'] == 'fruit_5_5' or config['data_set'] == 'dog_5_5':
        config['network']['params']['class_num'] = 10
        config['network']['params']['shared_class_num'] = 5
    else:
        raise ValueError('dataset %s not found!' % (config['data_set']))

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
        os.makedirs(config['path']['log'])
        os.makedirs(config['path']['model'])
    
    config['logger'] = Logger(logroot=config['path']['log'], filename=config['task'], level='debug')
    config['logger'].logger.debug(str(config))
        
    empty_dict(config)
    config['results']['best'].append(train(config))
    print_dict(config)

