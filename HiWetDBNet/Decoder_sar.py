from ast import arg
import time
import numpy as np
import tifffile as tiff
from osgeo import gdal
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model.sar_branch import SARBranch
from Decoder_optical_config import decoder_args


def decoder(timeseries_image_path_list, model, device, predict_save_path, image_num):
    model.eval()
    dist_border = int((15 - 1) / 2)
    sar_image = tiff.imread(timeseries_image_path_list[0])
    row_num = sar_image.shape[1]
    col_num = sar_image.shape[2]
    del sar_image

    predict_image_s2 = np.zeros(shape=(row_num, col_num))
    predict_image_s3 = np.zeros(shape=(row_num, col_num))

    for row in range(row_num):
        for_start_time = time.time()
        print("==========开始执行第{}/{}行预测==========".format(row + 1, row_num))
        row += dist_border
        sar_patches = []
        for index in range(len(timeseries_image_path_list)):
            if index == 0:
                sar_input_image_path = timeseries_image_path_list[index]
                sar_image = tiff.imread(sar_input_image_path).astype(np.float32)
                sar_image_pad = np.zeros(shape=(
                    sar_image.shape[0], sar_image.shape[1] + 2 * dist_border, sar_image.shape[2] + 2 * dist_border))
                sar_image_pad[:, dist_border:sar_image_pad.shape[1] - dist_border,
                dist_border:sar_image_pad.shape[2] - dist_border] = sar_image

                del sar_image

                for col in range(col_num):
                    col += dist_border
                    sar_patch = sar_image_pad[:, row - dist_border:row + dist_border + 1,
                                col - dist_border:col + dist_border + 1]
                    sar_patch = np.expand_dims(sar_patch, 0)
                    sar_patches.append(sar_patch)

                del sar_image_pad
            else:
                sar_input_image_path = timeseries_image_path_list[index]
                sar_image = tiff.imread(sar_input_image_path).astype(np.float32)
                sar_image_pad = np.zeros(shape=(
                    sar_image.shape[0], sar_image.shape[1] + 2 * dist_border, sar_image.shape[2] + 2 * dist_border))
                sar_image_pad[:, dist_border:sar_image_pad.shape[1] - dist_border,
                dist_border:sar_image_pad.shape[2] - dist_border] = sar_image

                del sar_image

                for col in range(col_num):
                    col += dist_border
                    sar_patch = sar_image_pad[:, row - dist_border:row + dist_border + 1,
                                col - dist_border:col + dist_border + 1]
                    sar_patch = np.expand_dims(sar_patch, 0)
                    num = col - dist_border
                    sar_patches[num] = np.concatenate((sar_patches[num], sar_patch), axis=0)

                del sar_image_pad

        batch_size = 48
        x = len(sar_patches) // batch_size
        pred_list_sar2, pred_list_sar3 = [], []
        for k in range(x + 1):
            if k == x:
                sar_patches_part = torch.as_tensor(np.array(sar_patches[batch_size * k:])).float().to(device)

                output_level2, _, output_level3, _ = model(sar_patches_part)

                probas_output3 = F.softmax(output_level3, dim=1)
                probas_output2 = F.softmax(output_level2, dim=1)
                _, pred_output3 = torch.max(probas_output3, dim=1)
                _, pred_output2 = torch.max(probas_output2, dim=1)
                pred_output3 = pred_output3.cpu().numpy()
                pred_output2 = pred_output2.cpu().numpy()

                pred_list_sar3.extend(pred_output3)
                pred_list_sar2.extend(pred_output2)

                del sar_patches_part

            else:
                sar_patches_part = torch.as_tensor(
                    np.array(sar_patches[batch_size * k:batch_size * (k + 1)])).float().to(device)

                output_level2, _, output_level3, _ = model(sar_patches_part)

                probas_output3 = F.softmax(output_level3, dim=1)
                probas_output2 = F.softmax(output_level2, dim=1)
                _, pred_output3 = torch.max(probas_output3, dim=1)
                _, pred_output2 = torch.max(probas_output2, dim=1)
                pred_output3 = pred_output3.cpu().numpy()
                pred_output2 = pred_output2.cpu().numpy()

                pred_list_sar3.extend(pred_output3)
                pred_list_sar2.extend(pred_output2)

                del sar_patches_part

            predict_image_s3[row - dist_border, :] = pred_list_sar3
            predict_image_s2[row - dist_border, :] = pred_list_sar2

            for_end_time = time.time()
            print("Execution of line {}/{} prediction time: {:.4f}s".format(row - dist_border + 1, row_num,
                                                                            for_end_time - for_start_time))

        tiff.imwrite(predict_save_path + r'\sar_level3_{}.tif'.format(image_num), predict_image_s3)
        tiff.imwrite(predict_save_path + r'\sar_level2_{}.tif'.format(image_num), predict_image_s2)


if __name__ == "__main__":
    args = decoder_args.args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 1
    for i in range(epoch):
        print("========== The {}th model prediction ==========".format(i + 1))
        start_time = time.time()

        predict_save_path = args['predict_save_path'][i]
        sar_image_path = args['sar_image_path'][i]
        model_path = args['model_path'][i]
        input_dim = args['input_dim'][i]
        hidden_dim = args['hidden_dim'][i]
        num_class2 = args['num_class2'][i]
        num_class3 = args['num_class3'][i]
        kernel_size = args['kernel_size'][i]
        num_layers = args['num_layers'][i]
        batch_first = args['batch_first'][i]
        bias = args['bias'][i]
        return_all_layers = args['return_all_layers'][i]

        sar_image_path_list = [os.path.join(sar_image_path, item) for item in os.listdir(sar_image_path)]
        timeseries_image_path_list = []
        for image_path in sar_image_path_list:
            image_path_name, image_path_type = os.path.splitext(image_path)
            if image_path_type == r'.tif':
                timeseries_image_path_list.append(image_path)

        model = SARBranch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_class2=num_class2,
            num_class3=num_class3,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bias=bias,
            return_all_layers=return_all_layers
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        decoder(timeseries_image_path_list, model, device, predict_save_path, image_num=i + 1)

        end_time = time.time()
        print("The forecast time is: {:.4f}s".format(end_time - start_time))
        print("\n")
