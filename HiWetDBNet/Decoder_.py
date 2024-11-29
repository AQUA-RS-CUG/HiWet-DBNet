import time
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
import os
from model.mscnn import MSCNN
from Decoder_config import decoder_args


# time.sleep(7000)

def decoder(optical_image_path, timeseries_image_path_list, model, device, predict_save_path, image_num):
    model.eval()
    dist_border = int((15 - 1) / 2)
    optical_image = tiff.imread(optical_image_path)
    row_num = optical_image.shape[1]
    col_num = optical_image.shape[2]
    del optical_image

    predict_image_s2 = np.zeros(shape=(row_num, col_num))
    predict_image2 = np.zeros(shape=(row_num, col_num))
    predict_image_s3 = np.zeros(shape=(row_num, col_num))
    predict_image3 = np.zeros(shape=(row_num, col_num))

    for row in range(row_num):
        for_start_time = time.time()
        print("==========开始执行第{}/{}行预测==========".format(row + 1, row_num))
        row += dist_border
        optical_patches = []
        sar_patches = []
        for index in range(len(timeseries_image_path_list)):
            if index == 0:
                sar_input_image_path = timeseries_image_path_list[index]
                sar_image = tiff.imread(sar_input_image_path).astype(np.float32)
                optical_image = tiff.imread(optical_image_path).astype(np.float32)
                sar_image_pad = np.zeros(shape=(
                sar_image.shape[0], sar_image.shape[1] + 2 * dist_border, sar_image.shape[2] + 2 * dist_border))
                optical_image_pad = np.zeros(shape=(optical_image.shape[0], optical_image.shape[1] + 2 * dist_border,
                                                    optical_image.shape[2] + 2 * dist_border))
                sar_image_pad[:, dist_border:sar_image_pad.shape[1] - dist_border,
                dist_border:sar_image_pad.shape[2] - dist_border] = sar_image
                optical_image_pad[:, dist_border:optical_image_pad.shape[1] - dist_border,
                dist_border:optical_image_pad.shape[2] - dist_border] = optical_image

                del optical_image
                del sar_image

                for col in range(col_num):
                    col += dist_border
                    optical_patch = optical_image_pad[:, row - dist_border:row + dist_border + 1,
                                    col - dist_border:col + dist_border + 1]
                    sar_patch = sar_image_pad[:, row - dist_border:row + dist_border + 1,
                                col - dist_border:col + dist_border + 1]
                    sar_patch = np.expand_dims(sar_patch, 0)
                    optical_patches.append(optical_patch)
                    sar_patches.append(sar_patch)
                del optical_image_pad
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

        batch_size = 256  # Set the batch of models to be entered at one time
        x = len(sar_patches) // batch_size
        pred_list_sar3, pred_list_output3 = [], []
        pred_list_sar2, pred_list_output2 = [], []
        for k in range(x + 1):
            if k == x:
                optical_patches_part = torch.as_tensor(np.array(optical_patches[batch_size * k:])).float().to(device)
                sar_patches_part = torch.as_tensor(np.array(sar_patches[batch_size * k:])).float().to(device)
                # Predict
                output_level2, output_level3, s_output2, s_output3 = model(optical_patches_part, sar_patches_part)

                probas_output3 = F.softmax(output_level3, dim=1)
                probas_sar3 = F.softmax(s_output3, dim=1)
                probas_output2 = F.softmax(output_level2, dim=1)
                probas_sar2 = F.softmax(s_output2, dim=1)
                _, pred_output3 = torch.max(probas_output3, dim=1)
                _, pred_sar3 = torch.max(probas_sar3, dim=1)
                _, pred_output2 = torch.max(probas_output2, dim=1)
                _, pred_sar2 = torch.max(probas_sar2, dim=1)
                pred_output3 = pred_output3.cpu().numpy()
                pred_sar3 = pred_sar3.cpu().numpy()
                pred_output2 = pred_output2.cpu().numpy()
                pred_sar2 = pred_sar2.cpu().numpy()

                pred_list_sar3.extend(pred_sar3)
                pred_list_output3.extend(pred_output3)
                pred_list_sar2.extend(pred_sar2)
                pred_list_output2.extend(pred_output2)

                del optical_patches_part
                del sar_patches_part

            else:
                optical_patches_part = torch.as_tensor(
                    np.array(optical_patches[batch_size * k:batch_size * (k + 1)])).float().to(device)
                sar_patches_part = torch.as_tensor(
                    np.array(sar_patches[batch_size * k:batch_size * (k + 1)])).float().to(device)

                # Predict
                output_level2, output_level3, s_output2, s_output3 = model(optical_patches_part, sar_patches_part)

                probas_output3 = F.softmax(output_level3, dim=1)
                probas_sar3 = F.softmax(s_output3, dim=1)
                probas_output2 = F.softmax(output_level2, dim=1)
                probas_sar2 = F.softmax(s_output2, dim=1)
                _, pred_output3 = torch.max(probas_output3, dim=1)
                _, pred_sar3 = torch.max(probas_sar3, dim=1)
                _, pred_output2 = torch.max(probas_output2, dim=1)
                _, pred_sar2 = torch.max(probas_sar2, dim=1)
                pred_output3 = pred_output3.cpu().numpy()
                pred_sar3 = pred_sar3.cpu().numpy()
                pred_output2 = pred_output2.cpu().numpy()
                pred_sar2 = pred_sar2.cpu().numpy()

                pred_list_sar3.extend(pred_sar3)
                pred_list_output3.extend(pred_output3)
                pred_list_sar2.extend(pred_sar2)
                pred_list_output2.extend(pred_output2)

                del optical_patches_part
                del sar_patches_part

        # Assign the prediction result to the prediction image
        predict_image_s3[row - dist_border, :] = pred_list_sar3
        predict_image3[row - dist_border, :] = pred_list_output3
        predict_image_s2[row - dist_border, :] = pred_list_sar2
        predict_image2[row - dist_border, :] = pred_list_output2

        for_end_time = time.time()

        print("Execution of line {}/{} prediction time: {:.4f}s".format(row - dist_border + 1, row_num, for_end_time - for_start_time))

    tiff.imwrite(predict_save_path + r'\sar_level3_{}.tif'.format(image_num), predict_image_s3)
    tiff.imwrite(predict_save_path + r'\mscnn_level3_{}.tif'.format(image_num), predict_image3)
    tiff.imwrite(predict_save_path + r'\sar_level2_{}.tif'.format(image_num), predict_image_s2)
    tiff.imwrite(predict_save_path + r'\mscnn_level2_{}.tif'.format(image_num), predict_image2)


if __name__ == "__main__":
    args = decoder_args.args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 8
    for i in range(epoch):
        print("========== The {}th model prediction =========".format(i + 1))
        start_time = time.time()
        predict_save_path = args['predict_save_path'][i]
        optical_image_path = args['optical_image_path'][i]
        sar_image_path = args['sar_image_path'][i]
        model_path = args['model_path'][i]
        input_dim = args['input_dim'][i]
        hidden_dim = args['hidden_dim'][i]
        num_class2 = args['num_class2'][i]
        num_class3 = args['num_class3'][i]
        kernel_size = args['kernel_size'][i]
        num_layers = args['num_layers'][i]
        batch_first = args['batch_first'][i]
        return_all_layers = args['return_all_layers'][i]

        sar_image_path_list = [os.path.join(sar_image_path, item) for item in os.listdir(sar_image_path)]
        timeseries_image_path_list = []
        for image_path in sar_image_path_list:
            image_path_name, image_path_type = os.path.splitext(image_path)
            if image_path_type == r'.tif':
                timeseries_image_path_list.append(image_path)

        model = MSCNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_class2=num_class2,
            num_class3=num_class3,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bias=True,
            return_all_layers=return_all_layers
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        decoder(optical_image_path, timeseries_image_path_list, model, device, predict_save_path, image_num=i + 1)
        end_time = time.time()
        print("The forecast time is: {:.4f}s".format(end_time - start_time))
        print("\n")
