import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from functions import labels2cat, Dataset_CRNN, train, acc_calculate, validation
from model import CRNN

working = f'light/10'
train_image_path = f'../datasets/' + working + '/train/crop'
test_image_path = f'../datasets/' + working + '/test/crop'
train_mat_path = f'../datasets/' + working + '/train/Mat'
test_mat_path = f'../datasets/' + working + '/test/Mat'
n_frames = 32
input_type = 'both'
batch_size = 32
num_workers = 16

# 创建一个示例数据集
X = np.random.rand(100, 10)  # 100个样本，每个样本有10个特征

# 初始化t-SNE模型
tsne = TSNE(n_components=2, random_state=42)

# 对数据进行降维
X_tsne = tsne.fit_transform(X)

# 可视化结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#
#
# action_names = os.listdir(train_image_path)
#
# le = LabelEncoder()
# le.fit(action_names)
#
# # show how many classes
# print('labels:{}'.format(list(le.classes_)))
#
# # convert category -> 1-hot
# action_category = le.transform(action_names).reshape(-1, 1)
# enc = OneHotEncoder()
# enc.fit(action_category)
#
# train_actions = []
# train_all_names = []
# test_actions = []
# test_all_names = []
# for action in action_names:
#     for f_name in os.listdir(f'{train_image_path}/{action}'):
#         train_actions.append(action)
#         train_all_names.append(f'{action}/{f_name}')
#
#     for f_name in os.listdir(f'{test_image_path}/{action}'):
#         test_actions.append(action)
#         test_all_names.append(f'{action}/{f_name}')
#
# train_list = train_all_names
# train_label = labels2cat(le, train_actions)
# test_list = test_all_names  # all video file names
# test_label = labels2cat(le, test_actions)  # all video labels
#
# transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.5], std=[0.5])])  # 串联多个图片变换
#
# train_set = data.DataLoader(Dataset_CRNN(train_image_path, train_mat_path,
#                          train_list, train_label, n_frames, transform=transform, input_type=input_type))
# # test_set = Dataset_CRNN(test_image_path, test_mat_path,
# #                         test_list, test_label, n_frames, transform=transform, input_type=input_type)
#
# train_loader = data.DataLoader(train_set, batch_size=batch_size,
#                                shuffle=True, num_workers=num_workers)
# # test_loader = data.DataLoader(test_set, batch_size=batch_size,
# #                               shuffle=False, num_workers=num_workers)
#
# data_image = []
# data_csi = []
# target = []
# for batch_idx, (image, mat, label) in train_loader:
#     data_image.append(image)
#     data_csi.append(mat)
#     target.append(label)
# data_image = np.concatenate(data_image)
# data_csi = np.concatenate(data_csi)
# print(data_image)
# print(data_csi)