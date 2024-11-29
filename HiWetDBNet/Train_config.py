class args:
    # ============= train =============
    num_class_level2: int = 4
    num_class_level3: int = 7

    run_information_path = r'C:\Users\16862\Desktop\HiWetDBNet_upload\run_information'
    model_save_path = r'C:\Users\16862\Desktop\HiWetDBNet_upload\mode_save'

    optical_path: str = r'C:\Users\DELL\Desktop\Experiment Model\dataset\optical_time10_interval1_addIndex'
    sar_path: str = r'C:\Users\DELL\Desktop\Experiment Model\dataset\sar_v9_time10_interval1'

    optical_train_path: str = r''
    sar_train_path: str = ''
    optical_valid_path: str = r''
    sar_valid_path: str = ''
    optical_test_path: str = r''
    sar_test_path: str = ''

    epoch = 220
    batch_size = 96
    learning_rate = 1e-3
    test_split_pro = 0.2 # Training set