import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import time
from tqdm import tqdm
from Train_config import args
import random
import numpy as np
import matplotlib.pyplot as plt
from model.mscnn import MSCNN
from user_dataset import MSCNN_Dataset
from focal_loss import MultiFocalLoss

# def seed_torch(seed=3407):
#     seed = int(seed)
#     random.seed(seed) # random设置随机数
#     os.environ['PYTHONHASHSEED'] = str(seed) # 环境变量
#     np.random.seed(seed) # numpy固定随机种子
#     torch.manual_seed(seed) # CPU情况下的随机数生成种子
#     torch.cuda.manual_seed(seed) # 设置当前GPU的随机数生成种子
#     torch.cuda.manual_seed_all(seed) # 设置所有GPU的随机数生成种子
#     torch.backends.cudnn.deterministic = True # 确定卷积算法类型（默认算法）
#     torch.backends.cudnn.benchmark = False # cudnn使用确定性卷积，而不是使用优化提速型的卷积
#     torch.backends.cudnn.enabled = False # 不适用cudnn底层加速

# time.sleep(2000)

# seed_torch()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
__foldername__ = ['FLAT', 'GU', 'NDLW', 'SAND', 'SAV', 'TCYC', 'WATER']

# Get sample paths for different categories
optical_train_sample_path_list = [os.path.join(args.optical_path, item) for item in os.listdir(args.optical_path)]
sar_train_sample_path_list = [os.path.join(args.sar_path, item) for item in os.listdir(args.sar_path)]

# Optical dataset
optical_train_image_path_list = []  # training
optical_valid_image_path_list = []  # validation
# Radar dataset
sar_train_image_path_list = []  # training
sar_valid_image_path_list = []  # validation
# Label
train_label_list2, train_label_list3 = [], []  # training
valid_label_list2, valid_label_list3 = [], []  # validation

# Get samples and set level3 labels
index = 0
for i in range(len(__foldername__)):
    optical_image_path_list = [os.path.join(optical_train_sample_path_list[i], item) for item in
                               os.listdir(optical_train_sample_path_list[i])]
    sar_image_path_list = [os.path.join(sar_train_sample_path_list[i], item) for item in
                           os.listdir(sar_train_sample_path_list[i])]
    # optical_train_image_path_list.extend(random.sample(optical_image_path_list, int(len(optical_image_path_list) * (1 - args.test_split_pro))))
    # optical_valid_image_path_list.extend(random.sample(optical_image_path_list, int(len(optical_image_path_list) * args.test_split_pro)))
    # sar_train_image_path_list.extend(random.sample(sar_image_path_list, int(len(sar_image_path_list) * (1 - args.test_split_pro))))
    # sar_valid_image_path_list.extend(random.sample(sar_image_path_list, int(len(sar_image_path_list) * args.test_split_pro)))
    optical_train_image_path_list.extend(optical_image_path_list)
    sar_train_image_path_list.extend(sar_image_path_list)

    # assert int(len(optical_image_path_list) * (1 - args.test_split_pro)) == int(len(sar_image_path_list) * (1 - args.test_split_pro))
    # assert int(len(optical_image_path_list) * args.test_split_pro) == int(len(sar_image_path_list) * args.test_split_pro)
    # train_num = int(len(optical_image_path_list) * (1 - args.test_split_pro))
    # valid_num = int(len(optical_image_path_list) * args.test_split_pro)

    for j in range(len(optical_image_path_list)):
        train_label_list3.append(index)

    # for k in range(valid_num):
    #     valid_label_list3.append(index)

    index += 1

# Set the labels of level2 according to the labels of level3
for i in train_label_list3:
    if i == 0:
        train_label_list2.append(0)
    if i == 1:
        train_label_list2.append(1)
    if i == 2:
        train_label_list2.append(1)
    if i == 3:
        train_label_list2.append(3)
    if i == 4:
        train_label_list2.append(1)
    if i == 5:
        train_label_list2.append(1)
    if i == 6:
        train_label_list2.append(2)

# Old version===
train_sample = list(zip(optical_train_image_path_list, sar_train_image_path_list, train_label_list2, train_label_list3))
random.shuffle(train_sample)
optical_train_image_path_list[:], sar_train_image_path_list[:], train_label_list2[:], train_label_list3[:] = zip(
    *train_sample)
train_dataset = MSCNN_Dataset(optical_train_image_path_list, sar_train_image_path_list, train_label_list2,
                              train_label_list3)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)

# New version===
# train_sample = list(zip(optical_train_image_path_list, sar_train_image_path_list, train_label_list2, train_label_list3))
# random.shuffle(train_sample)
# optical_train_image_path_list, sar_train_image_path_list, train_label_list2, train_label_list3 = zip(*train_sample)
# train_dataset = MSCNN_Dataset(optical_train_image_path_list, sar_train_image_path_list, train_label_list2, train_label_list3)
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)
# valid_sample = list(zip(optical_valid_image_path_list, sar_valid_image_path_list, valid_label_list2, valid_label_list3))
# random.shuffle(valid_sample)
# optical_valid_image_path_list, sar_valid_image_path_list, valid_label_list2, valid_label_list3 = zip(*valid_sample)
# valid_dataset = MSCNN_Dataset(optical_valid_image_path_list, sar_valid_image_path_list, valid_label_list2, valid_label_list3)
# valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)

# Model Instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSCNN(input_dim=4,
              hidden_dim=[64, 128],
              num_class2=args.num_class_level2,
              num_class3=args.num_class_level3,
              kernel_size=(3, 3),
              num_layers=2,
              batch_first=True,
              bias=True,
              return_all_layers=True
              )

model.to(device)
criterion_level3 = MultiFocalLoss(num_class=7).to(device)
criterion_level2 = MultiFocalLoss(num_class=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0006)

all_train_loss_list = []
train_loss_list2, train_loss_list3 = [], []
s_train_loss_list2, s_train_loss_list3 = [], []
s_correct_list2, s_correct_list3 = [], []
all_correct_list2, all_correct_list3 = [], []
epoch_list = []
# Cosine annealing algorithm to adjust learning rate
lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=4, eta_min=0)

print('>>>>>>>>>>  训练开始  <<<<<<<<<<')
for epoch in range(1, args.epoch + 1):
    start_time = time.time()
    model.train()
    avg_loss2, avg_loss3 = 0, 0
    all_avg_loss = 0
    s_avg_loss2, s_avg_loss3 = 0, 0
    cnt = 0
    print("Epoch {}/{}".format(epoch, args.epoch))
    print("learning rate: {}".format(optimizer.param_groups[0]["lr"]))

    for i, batch in tqdm(enumerate(train_dataloader)):
        optical_train_batch = batch['optical_image'].to(device)
        sar_train_batch = batch['sar_image'].to(device)
        label_train_batch2 = batch['label2'].to(device).long()
        label_train_batch3 = batch['label3'].to(device).long()

        optimizer.zero_grad()
        output_level2, output_level3, s_level2, s_level3 = model(optical_train_batch, sar_train_batch)

        train_loss2 = criterion_level2(output_level2, label_train_batch2)
        train_loss3 = criterion_level3(output_level3, label_train_batch3)
        s_train_loss2 = criterion_level2(s_level2, label_train_batch2)
        s_train_loss3 = criterion_level3(s_level3, label_train_batch3)
        all_train_loss = train_loss2 + s_train_loss2 + train_loss3 + s_train_loss3

        all_avg_loss += all_train_loss.item()
        avg_loss2 += train_loss2.item()
        s_avg_loss2 += s_train_loss2.item()
        avg_loss3 += train_loss3.item()
        s_avg_loss3 += s_train_loss3.item()

        cnt += 1
        all_train_loss.backward()
        optimizer.step()
    if epoch < 25:  # 35
        lr_scheduler.step()
    # lr_scheduler.step()

    # model.eval()
    # with torch.no_grad():
    #     s_correct2, s_correct3 = 0, 0
    #     all_correct2, all_correct3 = 0, 0
    #     avg_valid_loss2, avg_valid_loss3 = 0, 0
    #     s_avg_valid_loss2, s_avg_valid_loss3 = 0, 0
    #     total = 0
    #     # iter_num = 0
    #     for i, batch in enumerate(valid_dataloader):
    #         optical_valid_batch = batch['optical_image'].to(device)
    #         sar_valid_batch = batch['sar_image'].to(device)
    #         label_valid_batch2 = batch['label2'].to(device).long()
    #         label_valid_batch3 = batch['label3'].to(device).long()

    #         output_level2, output_level3, s_level2, s_level3 = model(optical_valid_batch, sar_valid_batch)

    #         # iter_num += 1
    #         probas_output2 = F.softmax(output_level2, dim=1)
    #         probas_output3 = F.softmax(output_level3, dim=1)
    #         probas_sar2 = F.softmax(s_level2, dim=1)
    #         probas_sar3 = F.softmax(s_level3, dim=1)
    #         _, pred_output2 = torch.max(probas_output2, dim=1)
    #         _, pred_output3 = torch.max(probas_output3, dim=1)
    #         _, pred_sar2 = torch.max(probas_sar2, dim=1)
    #         _, pred_sar3 = torch.max(probas_sar3, dim=1)

    #         total += label_valid_batch3.size(0)

    #         all_correct2 += (pred_output2 == label_valid_batch2).sum().item()
    #         all_correct3 += (pred_output3 == label_valid_batch3).sum().item()
    #         s_correct2 += (pred_sar2 == label_valid_batch2).sum().item()
    #         s_correct3 += (pred_sar3 == label_valid_batch3).sum().item()

    #     all_correct_list2.append(all_correct2/total)
    #     all_correct_list3.append(all_correct3/total)
    #     s_correct_list2.append(s_correct2/total)
    #     s_correct_list3.append(s_correct3/total)

    end_time = time.time()

    all_train_loss_list.append(all_avg_loss / cnt)
    train_loss_list2.append(avg_loss2 / cnt)
    train_loss_list3.append(avg_loss3 / cnt)
    s_train_loss_list2.append(s_avg_loss2 / cnt)
    s_train_loss_list3.append(s_avg_loss3 / cnt)

    epoch_list.append(epoch)
    print('========== Training Info ==========')
    print("all avg_loss: {}".format(all_avg_loss / cnt))
    print("train loss2: {}, train_loss3:{}".format(avg_loss2 / cnt, avg_loss3 / cnt))
    print("s_train loss2: {}, s_train loss3:{}".format(s_avg_loss2 / cnt, s_avg_loss3 / cnt))
    # print('==========验证准确率==========')
    # print("s_correct2: {}, s_correct3: {}".format(s_correct2/total, s_correct3/total))
    # print("all_correct2: {}, all_correct3: {}".format(all_correct2/total, all_correct3/total))

    print("time:{}".format(end_time - start_time))
    print("\n")

# plt.figure(1)
# plt.plot(epoch_list, all_train_loss_list, label='total loss')
# plt.plot(epoch_list, train_loss_list2, label='train loss level2')
# plt.plot(epoch_list, train_loss_list3, label='train loss level3')

# plt.plot(epoch_list, s_train_loss_list2, label='s train loss level2')
# plt.plot(epoch_list, s_train_loss_list3, label='s train loss level3')
# plt.legend(loc='best')

# plt.figure(2)
# plt.plot(epoch_list, all_correct_list2, label='correct level2')
# plt.plot(epoch_list, all_correct_list3, label='correct level3')
# plt.plot(epoch_list, s_correct_list2, label='sar correct level2')
# plt.plot(epoch_list, s_correct_list3, label='sar correct level3')
# plt.legend(loc='best')

# plt.show()

# Test=====
# model.eval()
# with torch.no_grad():
#     all_correct2, all_correct3 = 0, 0
#     s_correct2, s_correct3 = 0, 0
#     total = 0
#     for i, batch in enumerate(valid_dataloader):
#         optical_valid_batch = batch['optical_image'].to(device)
#         sar_valid_batch = batch['sar_image'].to(device)
#         label_valid_batch2 = batch['label2'].to(device).long()
#         label_valid_batch3 = batch['label3'].to(device).long()

#         output_level2, output_level3, s_level2, s_level3 = model(optical_valid_batch, sar_valid_batch)

#         probas_output2 = F.softmax(output_level2, dim=1)
#         probas_output3 = F.softmax(output_level3, dim=1)
#         probas_sar2 = F.softmax(s_level2, dim=1)
#         probas_sar3 = F.softmax(s_level3, dim=1)
#         _, pred_output2 = torch.max(probas_output2, dim=1)
#         _, pred_output3 = torch.max(probas_output3, dim=1)
#         _, pred_sar2 = torch.max(probas_sar2, dim=1)
#         _, pred_sar3 = torch.max(probas_sar3, dim=1)

#         total += label_valid_batch3.size(0)

#         all_correct2 += (pred_output2 == label_valid_batch2).sum().item()
#         all_correct3 += (pred_output3 == label_valid_batch3).sum().item()
#         s_correct2 += (pred_sar2 == label_valid_batch2).sum().item()
#         s_correct3 += (pred_sar3 == label_valid_batch3).sum().item()

#     print("Test accuracy:")
#     print("all valid acc3:{}".format(all_correct3/total))
#     print("all valid acc2:{}".format(all_correct2/total))
#     print("all sar valid acc3:{}".format(s_correct3/total))
#     print("all sar valid acc2:{}".format(s_correct2/total))

#     np.save(r'C:\Users\DELL\Desktop\Experiment Model\acc\test_acc-5', [all_correct3/total, all_correct2/total, s_correct3/total, s_correct2/total])

# torch.save(model.state_dict(), args.model_save_path + r'\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_10images.pth')
# torch.save(model.state_dict(), args.model_save_path + r'\mscnn_Focalloss_lr1e-4_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex.pth')
torch.save(model.state_dict(), args.model_save_path + r'\model.pth')
