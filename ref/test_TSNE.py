import argparse
import datetime
import json
import os
import random
import re
import sys
import time
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import optimizer
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from dataset import CSIVideoDataset
from models import ABLSTM, CNN, CSIModel, para_buid_model_stack_conv2D_new
from models.Transformer import THAT
import torch.nn.functional as F
from Smooth_labeling import LabelSmoothingCrossEntropy


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('--model', type=str, default='csimodel',
                        choices=['svm', 'knn', 'cnn', 'ablstm',
                                 'transformer', 'csimodel', 'r2plus1d', 'buid_model_stack_conv2D_new',
                                 'para_buid_model_stack_conv2D_new'],
                        help='model')
    parser.add_argument('--number', type=str, default='',
                        help='number of training data')
    parser.add_argument('--data', type=str, default='csi',
                        help='data for training')
    parser.add_argument('--epoch', type=int, default=30,
                        help='number of training epoch(if "tune" is True, "epoch" will be 30)')
    parser.add_argument('-t', '--tune', action='store_true',
                        help='fine tune')
    parser.add_argument('--tdata', type=str, default='ivc', choices=['ivc', 'csi'],
                        help='data for fine tuning')
    parser.add_argument('--tepoch', type=int, default=50,
                        help='number of fine tuning epoch')
    parser.add_argument('--cuda', type=str, default='1',
                        help='cuda number')
    parser.add_argument('model_name', type=str,
                        help='results/trained_models/csi_models/file_name')
    parser.add_argument('best_epoch', type=int,
                        help='best epoch')
    args = parser.parse_args()

    if args.tune:
        args.epoch = 30
        if args.experiment in ['s1', 's2', 's3']:
            sys.exit('Only cross-domain experiments require fine-tuning!')
        if args.model == 'r2plus1d':
            sys.exit('Not within the scope of the experiment!')

    if args.model == 'r2plus1d':
        args.data = 'video'

    return args


def get_model(name, cuda, config=None):
    # print('1')
    if name == 'svm':
        return SVM(config)
    elif name == 'knn':
        return KNN(config)
    elif name == 'cnn':
        return CNN.ConvNet()
    elif name == 'ablstm':
        return ABLSTM.ABLSTM()
        # return ABLSTM.CSIModel()
    elif name == 'transformer':
        return THAT.HARTrans(cuda=cuda)
    elif name == 'csimodel':
        # print('2')
        return CSIModel.CSIModel(config)
    elif name == 'r2plus1d':
        model = R2Plus1D.R2Plus1D(config)
        # for name, value in model.named_parameters():
        #     if not re.match('layers.fc', name):
        #         value.requires_grad = False
        return model
    elif name == 'buid_model_stack_conv2D_new':
        return buid_model_stack_conv2D_new.Model(config)
    elif name == 'para_buid_model_stack_conv2D_new':
        return para_buid_model_stack_conv2D_new.Model()


def softmax_cross_entropy_loss(outputs, labels):
    log_value = torch.log(outputs)
    clamp_value = log_value
    cross_entropy = torch.mean(labels * clamp_value)
    return -cross_entropy


def self_softmax(logits, temperature=1):
    softmax_logits = torch.softmax(logits / float(temperature), dim=1)
    return softmax_logits


# def smooth_one_hot(true_labels: torch.Tensor, classes:int, smoothing = 0.1):
#     assert 0 <= smoothing <1
#     confidence = 1.0 - smoothing
#     label_shape = torch.Size((true_labels.size(0), classes))
#     with torch.no_grad():
#         true_dist = torch.empty(size = label_shape, device = true_labels.device)
#         true_dist.fill_(smoothing / (classes -1))
#         _, index = torch.max(true_labels, 1)
#         true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
#     return true_dist

def train_valid(experiment_type, num_epochs=30):
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练并验证模型
    print('Training model...')
    history = []
    best_acc = 0.0
    best_epoch = 0
    total_time = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        info = '[Epoch: {}/{}]\n'.format(epoch + 1, num_epochs)

        # 训练模型
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        train_data_size = 0
        valid_data_size = 0

        for inputs, labels in train_dataloader:
            # csi维度：[batch, 400, 90]
            # video维度：[batch, 3, 16, 112, 112]
            # labels维度：[batch]
            train_data_size += inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # # 前向传播
            outputs, _ = model(inputs)
            # outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计信息
            train_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            # 评价模型
            model.eval()

            for inputs, labels in valid_dataloader:
                valid_data_size += inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 前向传播
                outputs, _ = model(inputs)
                # outputs = model(inputs)
                loss = criterion(outputs, labels)
                # 统计信息
                valid_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(
                    labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss,
                        avg_train_acc, avg_valid_acc])

        # 更新最佳模型的epoch
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()
        total_time += epoch_end - epoch_start

        info += ('\tTraining: Loss: {:.4f}, Accuracy: {:.4f}%\n'
                 '\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%\n\tTime: {:.4f}s\n'
                 '\tBest Accuracy for validation : {:.4f}%, at epoch {:03d}\n').format(
            avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                            epoch_end - epoch_start, best_acc * 100, best_epoch)
        print(info)

        if data_type == 'csi':
            torch.save(model, os.path.join(
                csi_model_path, str(epoch + 1) + '.pt'))
        else:
            save_path = os.path.join(video_model_path, experiment_type)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.save(model, os.path.join(save_path, str(epoch + 1) + '.pt'))

        with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
            file.write(info)

    with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
        file.write(str(total_time))
        file.write('/')

    print('Done.\n')
    return history, best_epoch


def draw_figure(history):
    print('Drawing pictures...')
    history = np.array(history)

    plt.figure(1)
    plt.plot(history[:, 0:2])
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.savefig(os.path.join(results_path, 'loss_results.png'))

    plt.figure(2)
    plt.plot(history[:, 2:4])
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(results_path, 'accuracy_results.png'))
    print('Done.\n')

def tsne(data, target):
    print("TSNE-Begin")
    X_tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
    print("TSNE-Finish")
    print(X_tsne)
    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    m = {0: 'r', 1: 'y', 2: 'g', 3: 'b', 4: 'k', 5: 'm', 6: 'c'}
    t = []
    for v in target:
        t.append(m[int(v)])
    plt.figure(figsize=(13, 6.5))  # 设置画布大小
    plt.subplot(121)
    plt.yticks(fontproperties='Arial', size=16, weight='bold')  # 设置大小及加粗
    plt.xticks(fontproperties='Arial', size=16, weight='bold')
    # 取消每一个的边框
    # ax1 = plt.subplot(2, 3, 1)
    # plt.spines['right'].set_visible(False)  # 右边
    # ax1.spines['top'].set_visible(False)  # 上边

    # 图例
    # j = 0
    # class_list = config['class_list']
    # past_label = ''
    # for i in X_tsne:
    #     if(past_label!=class_list[int(target[j])]):
    #         plt.scatter(i[0], i[1], c=t[j], label=class_list[int(target[j])])
    #         past_label = class_list[int(target[j])]
    #     else:
    #         plt.scatter(i[0], i[1], c=t[j])
    #     j = j+1
    # plt.legend(loc='upper center', bbox_to_anchor=(1.9, 0.8), shadow=True, ncol=1,
    #            prop={"family": "Arial", "size": 16, 'weight': 'bold'})

    # 无图例
    plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=t)

    plt.savefig('images/tsne_已训练缺少smooth.png', dpi=120)
    plt.show()

def test(best_epoch, dtype, experiment_type):
    print('Testing model...')
    y_test = []
    y_pred = []
    data = []
    target = []
    logits = [[] for _ in range(7)]

    start = time.time()

    if dtype == 'csi':
        if torch.cuda.is_available() == False:
            best_model = torch.load(os.path.join(
                'results/trained_models/csi_models', args.model_name, str(best_epoch) + '.pt'),
                map_location=torch.device('cpu')).to(device)
        else:
            best_model = torch.load(os.path.join(
                'results/trained_models/csi_models', args.model_name, str(best_epoch) + '.pt')).to(device)
        best_model.eval()
        for csi, labels in test_dataloader:
            csi = csi.to(device)
            labels = labels.to(device)
            # outputs, _ = best_model(csi)
            outputs = best_model(csi)
            data.append(torch.flatten(outputs).detach().cpu().numpy())
            target.append(torch.flatten(labels).detach().cpu().numpy())
            outputs = torch.argmax(outputs, dim=1)
            y_test.append(labels.detach().cpu().numpy())
            y_pred.append(outputs.detach().cpu().numpy())
    else:
        best_model = torch.load(os.path.join(
            video_model_path, experiment_type, str(best_epoch) + '.pt')).to(device)
        best_model.eval()
        for videos, labels in test_dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs, _ = best_model(videos)
            # for i in range(batch):
            #     truth = labels[i].detach().cpu()
            #     prob = outputs[i].detach().cpu().numpy()
            #     logits[truth].append(prob)

            outputs = torch.argmax(outputs, dim=1)
            y_test.append(labels.detach().cpu().numpy())
            y_pred.append(outputs.detach().cpu().numpy())

    end = time.time()
    test_time = end - start
    with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
        file.write('test_total_time:')
        file.write(str(test_time))

    # 画TSNE图
    # if (len(data[0].shape) > 2):
    #     data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    print(len(data))
    print(data[0].shape)
    print(len(target))
    print(target[0].shape)
    tsne(data, target)


    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    # print(y_pred.shape)
    # print(y_test.shape)

    # print(y_pred.shape)
    con_mat = pd.DataFrame(data=confusion_matrix(
        y_test, y_pred), index=class_list, columns=class_list)
    con_mat_percent = pd.DataFrame(data=confusion_matrix(
        y_test, y_pred, normalize='pred'), index=class_list, columns=class_list)
    con_mat.to_csv(os.path.join(results_path, 'con_mat.csv'))
    con_mat_percent.to_csv(os.path.join(results_path, 'con_mat_percent.csv'))

    report = classification_report(y_test, y_pred, digits=4)
    with open(os.path.join(results_path, 'test_report.txt'), 'a') as file:
        file.write(report)

    plt.figure(num=3, figsize=(num_classes, num_classes))
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='Blues')
    # sns.heatmap(con_mat, annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(results_path, 'confusion_matrix.png'))

    print('Done.\n')

    # 测试用
    # np.savez_compressed(experiment_type + '.npz', np.array(logits))


if __name__ == '__main__':
    # 配置参数
    args = get_args()
    set_random_seed(0)
    now = datetime.datetime.now()
    now = now.strftime('%Y_%m_%d_%H_%M_%S')
    with open('config.json', 'r') as file:
        config = json.load(file)

    data_type = 'video' if args.model == 'r2plus1d' else 'csi'

    data_num = 'train_' + args.number if args.number else 'train'
    # # 新增
    valid_num = 'valid_' + args.number if args.number else 'valid'
    test_num = 'test_' + args.number if args.number else 'test'

    data_path = os.path.join(config['output_path'], args.experiment)
    results_path = os.path.join(
        config['results_path'], args.experiment, args.model, now)
    csi_model_path = os.path.join(config['csi_model_path'], now)
    video_model_path = config['video_model_path']
    num_classes = config['num_classes']
    class_list = config['class_list']
    batch = config['batch_size']
    learning_rate = config['learning_rate']
    tune_learning_rate = config['tune_learning_rate']
    T = config['T']
    a = config['a']
    device = torch.device('cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')
    if args.tune:
        target_video_model = args.experiment.split('-')[1]
        video_model = os.path.join(
            video_model_path, target_video_model, 'feature.pt')
    else:
        video_model = os.path.join(
            video_model_path, args.experiment, 'feature.pt')

    info = ('Experiment info:\n[experiment]\t{}\n[model]\t\t{}\n[number]\t{}\n'
            '[data]\t\t{}\n[epoch]\t\t{}\n[fine tune]\t{}\n').format(
        args.experiment, args.model, data_num, args.data, args.epoch, args.tune)
    if args.tune:
        info += '[tune data type]{}\n[tune epoch]\t{}\n'.format(
            args.tdata, args.tepoch)
    info += 'OK :)\n\n'

    os.makedirs(results_path)
    os.makedirs(csi_model_path)
    with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
        file.write(info)

    print(info)

    # 读取数据集
    print('Loading data...')
    # train_dataloader = DataLoader(CSIVideoDataset(
    #     data_path=data_path, split=data_num, dtype=args.data), batch_size=batch, shuffle=True)
    # valid_dataloader = DataLoader(CSIVideoDataset(
    #     data_path=data_path, split=valid_num, dtype=data_type), batch_size=batch)
    test_dataloader = DataLoader(CSIVideoDataset(
        data_path=data_path, split=test_num, dtype=data_type), batch_size=1)
    if args.tune:
        tune_dataloader = DataLoader(CSIVideoDataset(
            data_path=data_path, split='tune_20', dtype=args.tdata), batch_size=batch, shuffle=True)
    print('Done.\n')

    # 创建模型并训练
    model = get_model(args.model, args.cuda, config).to(device)
    #history, best_epoch = train_valid(experiment_type=args.experiment, num_epochs=args.epoch)

    best_epoch = args.best_epoch

    # 测试模型并绘图
    # draw_figure(history)

    test(best_epoch, dtype=data_type,
         experiment_type=args.experiment)
