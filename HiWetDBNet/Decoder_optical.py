import time
import numpy as np
import tifffile as tiff
from osgeo import gdal
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model.optical_branch import OpticalBranch
from Decoder_optical_config import decoder_args

# time.sleep()

def decoder(optical_image_path, model, device, predict_save_path, image_num):
    model.eval()
    dist_border = int((15 - 1) / 2)
    optical_image = tiff.imread(optical_image_path)
    row_num = optical_image.shape[1]
    col_num = optical_image.shape[2]
    del optical_image

    predict_image2 = np.zeros(shape=(row_num, col_num))
    predict_image3 = np.zeros(shape=(row_num, col_num))

    for row in range(row_num):
        for_start_time = time.time()
        print("==========Begin execution of line {}/{} prediction==========".format(row+1, row_num))
        row += dist_border
        optical_patches = []
        optical_image = tiff.imread(optical_image_path).astype(np.float32)
        optical_image_pad = np.zeros(shape=(optical_image.shape[0], optical_image.shape[1]+2*dist_border, optical_image.shape[2]+2*dist_border))
        optical_image_pad[:, dist_border:optical_image_pad.shape[1]-dist_border, dist_border:optical_image_pad.shape[2]-dist_border] = optical_image
        
        del optical_image
        
        for col in range(col_num):
            col += dist_border
            optical_patch = optical_image_pad[:, row-dist_border:row+dist_border+1, col-dist_border:col+dist_border+1]
            optical_patches.append(optical_patch)
            
        del optical_image_pad
        
        batch_size = 48
        x = len(optical_patches) // batch_size
        pred_list_optical2, pred_list_optical3 = [], []
        for k in range(x + 1):
            if k == x:
                optical_patches_part = torch.as_tensor(np.array(optical_patches[batch_size*k:])).float().to(device)
                
                output_level2, _, output_level3, _ = model(optical_patches_part)
                
                probas_output3 = F.softmax(output_level3, dim=1)
                probas_output2 = F.softmax(output_level2, dim=1)
                _, pred_output3 = torch.max(probas_output3, dim=1)
                _, pred_output2 = torch.max(probas_output2, dim=1)
                pred_output3 = pred_output3.cpu().numpy()
                pred_output2 = pred_output2.cpu().numpy()
                
                pred_list_optical3.extend(pred_output3)
                pred_list_optical2.extend(pred_output2)
                
                del optical_patches_part
                
            else:
                optical_patches_part = torch.as_tensor(np.array(optical_patches[batch_size*k:batch_size*(k+1)])).float().to(device)
                
                output_level2, _, output_level3, _ = model(optical_patches_part)
                
                probas_output3 = F.softmax(output_level3, dim=1)
                probas_output2 = F.softmax(output_level2, dim=1)
                _, pred_output3 = torch.max(probas_output3, dim=1)
                _, pred_output2 = torch.max(probas_output2, dim=1)
                pred_output3 = pred_output3.cpu().numpy()
                pred_output2 = pred_output2.cpu().numpy()
                
                pred_list_optical3.extend(pred_output3)
                pred_list_optical2.extend(pred_output2)
                
                del optical_patches_part
                
        predict_image3[row-dist_border, :] = pred_list_optical3
        predict_image2[row-dist_border, :] = pred_list_optical2
            
        for_end_time = time.time()
            
        print("Predicted time to perform line {}/{}: {:.4f}s".format(row-dist_border+1, row_num, for_end_time-for_start_time))
            
        tiff.imwrite(predict_save_path + r'\optical_level3_{}.tif'.format(image_num), predict_image3)
        tiff.imwrite(predict_save_path + r'\optical_level2_{}.tif'.format(image_num), predict_image2)
            
if __name__ == "__main__":
    args = decoder_args.args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 1
    for i in range(epoch):
        print("========== The {}th model prediction ==========".format(i+1))
        start_time = time.time()
        
        predict_save_path = args['predict_save_path'][i]
        optical_image_path = args['optical_image_path'][i]
        model_path = args['model_path'][i]
        num_class2 = args['num_class2'][i]
        num_class3 = args['num_class3'][i]
        
        model = OpticalBranch(
            num_class2=num_class2,
            num_class3=num_class3
        )
        
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        decoder(optical_image_path, model, device, predict_save_path, image_num=i+1)
        
        end_time = time.time()
        print("The forecast time is: {:.4f}s".format(end_time-start_time))
        print('\n')