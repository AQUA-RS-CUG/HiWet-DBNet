class decoder_args:
    args = {
        # Prediction Result Save Path
        'predict_save_path': [
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\predict_addIndex\patch_1',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\predict_addIndex\patch_2',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\predict_addIndex\patch_3',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\predict_addIndex\patch_4',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\predict_addIndex\patch_5',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\predict_addIndex\patch_6',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\predict_addIndex\patch_7',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\predict_addIndex\patch_8',
        ],

        # Optical image path
        'optical_image_path': [
            r'C:\Users\DELL\Desktop\Sentinel-2\20210501\20210501_subset_layerstacking_patch_1.tif',
            r'C:\Users\DELL\Desktop\Sentinel-2\20210501\20210501_subset_layerstacking_patch_2.tif',
            r'C:\Users\DELL\Desktop\Sentinel-2\20210501\20210501_subset_layerstacking_patch_3.tif',
            r'C:\Users\DELL\Desktop\Sentinel-2\20210501\20210501_subset_layerstacking_patch_4.tif',
            r'C:\Users\DELL\Desktop\Sentinel-2\20210501\20210501_subset_layerstacking_patch_5.tif',
            r'C:\Users\DELL\Desktop\Sentinel-2\20210501\20210501_subset_layerstacking_patch_6.tif',
            r'C:\Users\DELL\Desktop\Sentinel-2\20210501\20210501_subset_layerstacking_patch_7.tif',
            r'C:\Users\DELL\Desktop\Sentinel-2\20210501\20210501_subset_layerstacking_patch_8.tif',
        ],

        # Radar image path
        'sar_image_path': [
            r'C:\Users\DELL\Desktop\Sentinel-1\20210501\patch_1',
            r'C:\Users\DELL\Desktop\Sentinel-1\20210501\patch_2',
            r'C:\Users\DELL\Desktop\Sentinel-1\20210501\patch_3',
            r'C:\Users\DELL\Desktop\Sentinel-1\20210501\patch_4',
            r'C:\Users\DELL\Desktop\Sentinel-1\20210501\patch_5',
            r'C:\Users\DELL\Desktop\Sentinel-1\20210501\patch_6',
            r'C:\Users\DELL\Desktop\Sentinel-1\20210501\patch_7',
            r'C:\Users\DELL\Desktop\Sentinel-1\20210501\patch_8',
        ],

        # Model save path
        'model_path': [
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\optical_focalloss_lr1e-3_epoch155.pth', # Optical Branch
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\optical_focalloss_lr1e-3_epoch155.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\optical_focalloss_lr1e-3_epoch155.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\optical_focalloss_lr1e-3_epoch155.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\optical_focalloss_lr1e-3_epoch155.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\optical_focalloss_lr1e-3_epoch155.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\optical_focalloss_lr1e-3_epoch155.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\optical_focalloss_lr1e-3_epoch155.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\test\sar_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_ablation.pth', # Radar Branch
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\test\sar_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_ablation.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\test\sar_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_ablation.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\test\sar_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_ablation.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\test\sar_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_ablation.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\test\sar_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_ablation.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\test\sar_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_ablation.pth',
            # r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\test\sar_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_ablation.pth',
            # Joint Classification
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_sar_time15.pth',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_sar_time15.pth',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_sar_time15.pth',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_sar_time15.pth',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_sar_time15.pth',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_sar_time15.pth',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_sar_time15.pth',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save\mscnn_Focalloss_lr1e-3_epoch155_removeavgpool_sarBranchAdjust_datasetAddIndex_sar_time15.pth',
        ],

        'data_process_path': [
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_process',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_process',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_process',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_process',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_process',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_process',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_process',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_process',
        ],

        'data_dataset_path': [
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
            r'C:\Users\16862\Desktop\HiWetDBNet_upload\process_file\data_dataset',
        ],

        # Hyperparameters
        'input_dim': [
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ],

        'hidden_dim': [
            [64, 128],
            [64, 128],
            [64, 128],
            [64, 128],
            [64, 128],
            [64, 128],
            [64, 128],
            [64, 128],
        ],

        'num_class2': [
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
        ],

        'num_class3': [
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
        ],

        'kernel_size': [
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
        ],

        'num_layers': [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ],

        'batch_first': [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],

        'return_all_layers': [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
    }
