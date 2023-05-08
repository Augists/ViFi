from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from dataset import CSIVideoDataset
import json
import numpy as np

print('Loading data...')
with open('config.json', 'r') as file:
    config = json.load(file)
data_path = os.path.join(config['output_path'], 's1')
batch = config['batch_size']
test = DataLoader(CSIVideoDataset(data_path=data_path, split='test', dtype='csi'))
data = []
target = []
for csi, labels in test:
    data.append(csi)
    target.append(labels)
print(len(data))
print(len(target))
data = np.concatenate(data)
if(len(data.shape)>2):
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
print("TSNE-Begin")
X_tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
print("TSNE-Finish")
print(X_tsne)
ckpt_dir="images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

m = {0: 'r', 1: 'y', 2: 'g', 3: 'b', 4: 'k', 5: 'm', 6: 'c'}
t = []
for v in target:
    t.append(m[int(v)])
plt.figure(figsize=(13, 6.5))    # 设置画布大小
plt.subplot(121)
plt.yticks(fontproperties='Arial', size=16, weight='bold')  # 设置大小及加粗
plt.xticks(fontproperties='Arial', size=16, weight='bold')
# 取消每一个的边框
# ax1 = plt.subplot(2, 3, 1)
# plt.spines['right'].set_visible(False)  # 右边
# ax1.spines['top'].set_visible(False)  # 上边

# 图例
j = 0
class_list = config['class_list']
past_label = ''
for i in X_tsne:
    if(past_label!=class_list[int(target[j])]):
        plt.scatter(i[0], i[1], c=t[j], label=class_list[int(target[j])])
        past_label = class_list[int(target[j])]
    else:
        plt.scatter(i[0], i[1], c=t[j])
    j = j+1
plt.legend(loc='upper center', bbox_to_anchor=(1.9, 0.8), shadow=True, ncol=1,
           prop={"family": "Arial", "size": 16, 'weight': 'bold'})

plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=t)

plt.savefig('images/tsne_未训练.png', dpi=120)
plt.show()

