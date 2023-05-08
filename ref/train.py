import argparse
import re
import datetime
import json
import os
import random
import sys
import time
import shutil

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

from dataset import CSIVideoDataset
from models import ABLSTM, CNN, CSIModel, R2Plus1D, R3D, C3D, TSNET, SACNN
from models.Transformer import THAT


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
                        choices=['cnn', 'ablstm', 'transformer', 'csimodel', 'sacnn',
                                 'r2plus1d', 'r3d', 'c3d', 'tsnet'],
                        help='model')
    parser.add_argument('--teacher', type=str, default='r2plus1d',
                        choices=['r2plus1d', 'c3d', 'r3d', 'tsnet'],
                        help='supervised model')
    parser.add_argument('--train_num', type=str, default='',
                        help='number of train data')
    parser.add_argument('--valid_num', type=str, default='',
                        help='number of valid data')
    parser.add_argument('--test_num', type=str, default='',
                        help='number of test data')
    parser.add_argument('--data', type=str, default='ivc',
                        choices=['ivc', 'csi', 'video'],
                        help='data for training')
    parser.add_argument('--epoch', type=int, default=30,
                        help='number of training epoch')
    parser.add_argument('--cuda', type=str, default='0',
                        help='cuda number')
    args = parser.parse_args()

    if args.model in ['r2plus1d', 'r3d', 'c3d', 'tsnet'] and args.data in ['ivc', 'csi'] \
        or args.model in ['cnn', 'ablstm', 'transformer', 'csimodel'] \
            and args.data == 'video':
        print("Model and data type mismatch!")
        sys.exit()

    return args


def get_model(name, cuda, config=None):
    if name == 'cnn':
        return CNN.ConvNet(config)
    elif name == 'ablstm':
        return ABLSTM.ABLSTM()
    elif name == 'transformer':
        return THAT.HARTrans(cuda=cuda)
    elif name == 'csimodel':
        return CSIModel.CSIModel(config)
    elif name == 'r2plus1d':
        return R2Plus1D.R2Plus1D(config)
    elif name == 'r3d':
        return R3D.R3D(config)
    elif name == 'c3d':
        return C3D.C3D(config)
    elif name == 'tsnet':
        return TSNET.TSNet(config)
    elif name == 'sacnn':
        return SACNN.SACNN(config)


def softmax_cross_entropy_loss(outputs, labels):
    log_value = torch.log(outputs)
    clamp_value = log_value
    cross_entropy = torch.mean(labels * clamp_value)
    return -cross_entropy


def self_softmax(logits, temperature=1):
    softmax_logits = torch.softmax(logits/float(temperature), dim=1)
    return softmax_logits


def purge(path, pattern):
    for f in os.listdir(path):
        if re.search(pattern, f):
            os.remove(os.path.join(path, f))


def train_valid(num_epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

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
            # 前向传播
            outputs, _ = model(inputs)
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
            torch.save(model, os.path.join(
                video_model_path, str(epoch + 1) + '.pt'))

        with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
            file.write(info)

    with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
        file.write('Train time: ' + str(total_time) + '\n')

    print('Done.\n')
    return history, best_epoch


def train_valid_ivc(num_epochs=30, T=5, a=0.4):
    student_model = model
    if os.path.exists(teacher_model_path):
        teacher_model = torch.load(teacher_model_path).to(device)
    else:
        print("Supervised model does not exist!")
        sys.exit()

    student_criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.KLDivLoss(reduction='sum')
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    print('Training model...')
    history = []
    best_acc = 0.0
    best_epoch = 0
    total_time = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        info = '[Epoch: {}/{}]\n'.format(epoch + 1, num_epochs)

        # 训练模型
        student_model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        train_data_size = 0
        valid_data_size = 0

        for videos, csi, labels in train_dataloader:
            # csi维度：[batch, 400, 90]
            # video维度：[batch, 3, 16, 112, 112]
            # labels维度：[batch]
            train_data_size += csi.size(0)
            videos = videos.float().to(device)
            csi = csi.to(device)
            labels = labels.to(device)
            # 前向传播
            s_outputs, s_feature = student_model(csi)
            t_outputs, t_feature = teacher_model(videos)

            loss_1 = distill_criterion(F.log_softmax(
                s_outputs / T, dim=1), F.softmax(t_outputs / T, dim=1))
            # loss_1 = softmax_cross_entropy_loss(self_softmax(
            #     s_outputs, T), F.softmax(t_outputs / T, dim=1))
            # loss_1 = nn.L1Loss()(s_outputs / T, t_outputs / T)
            # loss_1 = nn.MSELoss()(s_outputs / T, t_outputs / T)

            # loss_2 = nn.L1Loss()(s_feature / T, t_feature / T)
            loss_2 = nn.MSELoss()(s_feature / T, t_feature / T)

            loss = a * loss_1 + (1-a) * loss_2

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计信息
            train_loss += loss.item() * csi.size(0)
            _, predictions = torch.max(s_outputs.data, 1)
            correct_counts = predictions.eq(
                labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * csi.size(0)

        with torch.no_grad():
            # 评价模型
            student_model.eval()

            for inputs, labels in valid_dataloader:
                valid_data_size += inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 前向传播
                outputs, _ = student_model(inputs)
                loss = student_criterion(outputs, labels)
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

        torch.save(student_model, os.path.join(
            csi_model_path, str(epoch + 1) + '.pt'))

        with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
            file.write(info)

    with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
        file.write('Train time: ' + str(total_time) + '\n')

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


def test(best_epoch, dtype, model):
    print('Testing model...')
    y_test = []
    y_pred = []
    start = time.time()

    if dtype == 'csi':
        best_model = torch.load(os.path.join(
            csi_model_path, str(best_epoch) + '.pt')).to(device)
        best_model.eval()
        for csi, labels in test_dataloader:
            csi = csi.to(device)
            labels = labels.to(device)
            outputs, _ = best_model(csi)
            outputs = torch.argmax(outputs, dim=1)
            y_test.append(labels.detach().cpu().numpy())
            y_pred.append(outputs.detach().cpu().numpy())

        # 垃圾清理
        shutil.rmtree(csi_model_path)
    else:
        best_model = torch.load(os.path.join(
            video_model_path, str(best_epoch) + '.pt')).to(device)
        best_model.eval()
        for videos, labels in test_dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs, _ = best_model(videos)
            outputs = torch.argmax(outputs, dim=1)
            y_test.append(labels.detach().cpu().numpy())
            y_pred.append(outputs.detach().cpu().numpy())

        # 垃圾清理
        os.rename(os.path.join(video_model_path, str(best_epoch) + '.pt'),
                  os.path.join(video_model_path, model + '.pt'))
        purge(video_model_path, '[0-9].pt')

    end = time.time()
    test_time = end - start
    with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
        file.write('Test time: ' + str(test_time) + '\n')

    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    """ con_mat = pd.DataFrame(data=confusion_matrix(
        y_test, y_pred), index=class_list, columns=class_list) """
    # con_mat.to_csv(os.path.join(results_path, 'con_mat.csv'))
    report = classification_report(y_test, y_pred, digits=4)
    with open(os.path.join(results_path, 'test_report.txt'), 'a') as file:
        file.write(report)

    """ plt.figure(num=3, figsize=(num_classes, num_classes))
    sns.heatmap(con_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(results_path, 'confusion_matrix.png')) """

    con_mat_percent = pd.DataFrame(data=confusion_matrix(
    y_test, y_pred, normalize='pred'), index=class_list, columns=class_list)
    con_mat_percent.to_csv(os.path.join(results_path, 'con_mat_percent.csv'))
    plt.figure(num=4, figsize=(num_classes, num_classes))
    sns.heatmap(con_mat_percent, annot=True, fmt='.2%', cmap='Reds')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(results_path, 'confusion_matrix_percent.png'))
    print('Done.\n')


if __name__ == '__main__':
    #print('Hello')
    # 配置各项参数
    # set_random_seed(0)

    # 实验参数
    args = get_args()
    data_type = 'video' if args.data == 'video' else 'csi'
    train_num = 'train_' + args.train_num if args.train_num else 'train'
    valid_num = 'valid_' + args.valid_num if args.valid_num else 'valid'
    test_num = 'test_' + args.test_num if args.test_num else 'test'

    # 路径参数
    with open('config.json', 'r') as file:
        config = json.load(file)

    now = datetime.datetime.now()
    now = now.strftime('%Y_%m_%d_%H_%M_%S')
    data_path = os.path.join(config['output_path'], args.experiment)
    results_path = os.path.join(
        config['results_path'], args.experiment, args.model, now)
    csi_model_path = config['csi_model_path']
    video_model_path = os.path.join(
        config['video_model_path'], args.experiment)
    teacher = args.teacher
    teacher_model_path = os.path.join(video_model_path, teacher + '.pt')

    # 训练参数
    num_classes = config['num_classes']
    class_list = config['class_list']
    batch = config['batch_size']
    learning_rate = config['learning_rate']
    T = config['T']
    a = config['a']
    device = torch.device(
        'cuda:' + args.cuda if torch.cuda.is_available() else 'cpu')

    # 创建文件夹 打印日志
    info = ('Experiment info:\n[experiment]\t{}\n[model]\t\t{}\n'
            '[train number]\t{}\n[valid number]\t{}\n[test number]\t{}\n'
            '[data]\t\t{}\n[teacher]\t{}\n[epoch]\t\t{}\nOK :)\n\n').format(
                args.experiment, args.model, train_num, valid_num, test_num,
                args.data, args.teacher, args.epoch)
    print(info)
    os.makedirs(results_path)
    if not os.path.exists(csi_model_path) and data_type == 'csi':
        os.makedirs(csi_model_path)
    if not os.path.exists(video_model_path) and data_type == 'video':
        os.makedirs(video_model_path)

    with open(os.path.join(results_path, 'train_info.txt'), 'a') as file:
        file.write(info)

    # 读取数据集
    print('Loading data...')
    train_dataloader = DataLoader(CSIVideoDataset(
        data_path=data_path, split=train_num, dtype=args.data), batch_size=batch, shuffle=True)
    valid_dataloader = DataLoader(CSIVideoDataset(
        data_path=data_path, split=valid_num, dtype=data_type), batch_size=batch)
    test_dataloader = DataLoader(CSIVideoDataset(
        data_path=data_path, split=test_num, dtype=data_type), batch_size=batch)
    print('Done.\n')

    # 创建模型并训练
    model = get_model(args.model, args.cuda, config).to(device)
    if args.data == 'ivc':
        history, best_epoch = train_valid_ivc(num_epochs=args.epoch, T=T, a=a)
    else:
        history, best_epoch = train_valid(num_epochs=args.epoch)

    # 测试模型并绘图
    draw_figure(history)
    test(best_epoch, dtype=data_type, model=args.model)
