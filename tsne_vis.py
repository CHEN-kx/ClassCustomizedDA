import os
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pre_process as prep
from network import resnet50
from datasets import ImageList


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint1', default = './snapshot/home_10_10/model/rpf1_best_model.pth', help='checkpoint file')
    parser.add_argument('--checkpoint2', default = './snapshot/home_10_10/model/rpf1+f22_best_model.pth', help='checkpoint file')
    parser.add_argument('--checkpoint3', default = './snapshot/home_10_10/model/rp_best_model.pth', help='checkpoint file')
    parser.add_argument('--source_txt', default = './data/list/vis_home_R2P/source.txt')
    parser.add_argument('--target_unlabel_txt', default = './data/list/vis_home_R2P/target_unlabeled.txt')
    parser.add_argument('--target_label_txt', default = './data/list/vis_home_R2P/target_labeled.txt')
    parser.add_argument('--batch_size', type = int, default=36)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    
    # data prepare
    data_set = {}
    dset_loaders = {}
    params = {'resize_size': 256, 'crop_size': 224}

    prep_dict = {}
    prep_dict['test'] = prep.image_test(**params)
    data_set['s'] = ImageList(open(args.source_txt).readlines(), transform=prep_dict['test'])
    dset_loaders['s'] = torch.utils.data.DataLoader(data_set['s'],
                                                        batch_size=args.batch_size,
                                                        shuffle=False, num_workers=0)
    data_set['tl'] = ImageList(open(args.target_label_txt).readlines(), transform=prep_dict['test'])
    dset_loaders['tl'] = torch.utils.data.DataLoader(data_set['tl'],
                                                        batch_size=args.batch_size,
                                                        shuffle=False, num_workers=0)
    data_set['tu'] = ImageList(open(args.target_unlabel_txt).readlines(), transform=prep_dict['test'])
    dset_loaders['tu'] = torch.utils.data.DataLoader(data_set['tu'],
                                                        batch_size=args.batch_size,
                                                        shuffle=False, num_workers=0)
    

    model = resnet50().cuda()
    for i in range(3):
        print(i)
        results = []
        if i == 0:
            model.load_state_dict(torch.load(args.checkpoint1)['base'])
            save_fig = './tsne_f1_.png'
        elif i == 1:
            model.load_state_dict(torch.load(args.checkpoint2)['base'])
            save_fig = './tsne_f1+f2_.png'
        else:
            model.load_state_dict(torch.load(args.checkpoint3)['base'])
            save_fig = './tsne_all_.png'
        model.eval()
        
        with torch.no_grad():
            print('source')
            iter_test_s = iter(dset_loaders["s"])
            for i in range(len(dset_loaders['s'])):
                data_s = next(iter_test_s)
                inputs_s = data_s[0].cuda()     
                results.append(model(inputs_s))
            iter_test_tu = iter(dset_loaders["tu"])
            print('target unlabel')
            for i in range(len(dset_loaders['tu'])):
                data_tu = next(iter_test_tu)
                inputs_tu = data_tu[0].cuda()
                results.append(model(inputs_tu))
            iter_test_tl = iter(dset_loaders["tl"])
            print('target label')
            for i in range(len(dset_loaders['tl'])):
                data_tl = next(iter_test_tl)
                inputs_tl = data_tl[0].cuda()
                results.append(model(inputs_tl))
            
        embeddings = torch.cat([embedding for embedding in results], dim=0) # (num_samples, embed_dim)
        tsne = TSNE(n_components=2,init='pca', learning_rate=1000)
        embeddings_tsne= tsne.fit_transform(embeddings.cpu().numpy())

        lengths = 829
        lengthtu = 1382
        lengthtl = 10
        plt.cla()
        for k in range(0, lengths):#ECA8A9,D3E2B7,74AED4,fb8072,8dd3c7,8da0cb
            plt.scatter(embeddings_tsne[k, 0], embeddings_tsne[k, 1], c = '#8dd3c7', s = 6,marker='.')#red, cmap='viridis'
        for k in range(lengths, lengths+ lengthtu):
            plt.scatter(embeddings_tsne[k, 0], embeddings_tsne[k, 1], c = '#8da0cb', s = 6,marker='.')#yellow
        for k in range(lengths + lengthtu, lengths+ lengthtu+ lengthtl):
            plt.scatter(embeddings_tsne[k, 0], embeddings_tsne[k, 1], c = '#fb8072', s = 10,marker='*')#blue
        plt.savefig(save_fig, dpi=300)    
            
if __name__ == '__main__':
    main()
