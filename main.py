from __future__ import print_function
import os.path
from sklearn.model_selection import train_test_split
import os
import argparse
import glob
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40,shapeNetTest,shapeNetTrain, snc_synth_id_to_category
from model import GSNET, NewGSNET
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

folderTclass = {}

def _init_():
    np.random.seed(1125)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

    for idx,fd in enumerate(glob.glob('./data/shapenet/0*')):
        folderTclass[idx] = snc_synth_id_to_category[os.path.basename(fd)]
        

def train(args, io):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        pcd = np.empty((57449,2048,3),dtype=np.float32)
        label = np.empty((57449,),dtype=np.int64)
        i=0
        j=0
        for fd in glob.glob('./data/shapenet/0*'):
            for f in os.listdir(fd):
                pct = o3d.io.read_point_cloud(os.path.join(fd,f))
                pcd[i] = np.asarray(pct.points)
                label[i]=j
                i=i+1
            j=j+1

        if i != 57449 or j != 57:
            raise ValueError('data stat doesn\'t match')

        pcd_train, pcd_test, label_train, label_test = train_test_split(pcd, label, stratify=label,random_state=2215)

        train_loader = DataLoader(shapeNetTrain(p=pcd_train,label=label_train,num_points=args.num_points), num_workers=8, 
                                batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_loader = DataLoader(shapeNetTest(p=pcd_test,label=label_test,num_points=args.num_points), num_workers=8, 
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)


    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'GSNET':
        if args.dataset == 'shapenet':
            model = GSNET(args, output_channels=57).to(device)
        else:
            model = GSNET(args).to(device)
    elif args.model == 'NewGSNET':
        print('NewGSNET!')
        if args.dataset == 'shapenet':
            model = NewGSNET(args, output_channels=57).to(device)
        else:
            model = NewGSNET(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logfile.write(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        logfile.write("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        logfile.write("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=0.00001)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        logfile.write(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        logfile.write(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            io.cprint('Max Acc:%.6f' % best_test_acc)
            logfile.write('\nMax Acc:%.6f' % best_test_acc)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):

    if args.dataset == 'modelnet40':
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    else:
        pcd = np.empty((57449,2048,3),dtype=np.float32)
        label = np.empty((57449,),dtype=np.int64)
        i=0
        j=0
        for fd in glob.glob('./data/shapenet/0*'):
            for f in os.listdir(fd):
                pct = o3d.io.read_point_cloud(os.path.join(fd,f))
                pcd[i] = np.asarray(pct.points)
                label[i]=j
                i=i+1
            j=j+1

        if i != 57449 or j != 57:
            raise ValueError('data stat doesn\'t match')

        pcd_train, pcd_test, label_train, label_test = train_test_split(pcd, label, stratify=label, random_state=1152)

        test_loader = DataLoader(shapeNetTest(p=pcd_test,label=label_test,num_points=args.num_points), num_workers=8, 
                                batch_size=args.test_batch_size, shuffle=True, drop_last=False)


    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    # if args.dataset == 'modelnet40':
    #     model = GSNET(args).to(device)
    # else:
    #     model = GSNET(args,output_channels=57).to(device)

    if args.model == 'GSNET':
        if args.dataset == 'shapenet':
            model = GSNET(args, output_channels=57).to(device)
        else:
            model = GSNET(args).to(device)
    elif args.model == 'NewGSNET':
        print('NewGSNET!')
        if args.dataset == 'shapenet':
            model = NewGSNET(args, output_channels=57).to(device)
        else:
            model = NewGSNET(args).to(device)
    else:
        raise Exception("Not implemented")


    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    cnt = 0
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        if cnt <= 25:
            for index,inq in enumerate(preds.detach().cpu().numpy() != label.cpu().numpy()):

                if inq:
                    pcd = o3d.geometry.PointCloud()
                    # print(data.cpu().permute(0,2,1).numpy()[index].shape)
                    pcd.points = o3d.utility.Vector3dVector((data.cpu().permute(0,2,1).numpy())[index])
                    o3d.io.write_point_cloud("./errors/gsnet/{:d}-{:s}-{:s}.ply".format(cnt,
                        folderTclass[(preds.detach().cpu().numpy())[index]],
                        folderTclass[(label.cpu().numpy())[index]]),
                        pcd
                    )
                    cnt=cnt+1
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print(np.sum(test_true!=test_pred))
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)
    logfile.write(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='GSNET', metavar='N',
                        choices=['GSNET', 'NewGSNET'],
                        help='Model to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'shapenet'], help='dataset to be used, [modelnet40, shapenet]')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logfile = open(os.path.join(BASE_DIR, 'log.txt'),'w')


    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    logfile.write(str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
        logfile.write('USE GPU')
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
