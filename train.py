#-------------------------------------#
#       Train the data set
#-------------------------------------#
import datetime
import os

import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

'''
Training your own object detection model must pay attention to the following points:
1. Before training, carefully check whether your format meets the requirements. The library requires the format of the data set to be VOC format, and the content that needs to be prepared has input pictures and labels
The input image is a.jpg image, no need to fix the size, it will automatically resize before passing in the training.
The grayscale image will be automatically converted to RGB image for training, no need to modify yourself.
If the suffix of the input image is not jpg, you need to batch convert it to jpg before starting training.

The label is in.xml format and contains the target information to be detected. The label file corresponds to the input image file.

2. The loss value during training will be saved in the loss_%Y_%m_%d_%H_%M_%S folder in the logs folder

3. The trained weight file is saved in the logs folder.
'''

if __name__ == "__main__":
    #---------------------------------#
    # Cuda Specifies whether to use Cuda
    # No GPU can be set to False
    #---------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    # distributed:  Specifies whether to use standalone multi-card distributed operation
    # Terminal instruction supports Ubuntu only. CUDA_VISIBLE_DEVICES is used to specify the graphics card under Ubuntu.
    # In Windows system, DP mode is used by default to call all graphics cards, and DDP is not supported.
    # DP mode:
    # Set distributed = False
    # In the terminal type CUDA_VISIBLE_DEVICES=0,1 python train.py
    # DDP Mode:
    # Set distributed = True
    # In the terminal, enter CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn:  If sync_bn is used, the DDP mode is available for multiple cards
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training
    #   Can reduce the memory by about half, requires pytorch1.7.1 or more
    #---------------------------------------------------------------------#
    fp16            = True
    #---------------------------------------------------------------------#
    #   classes_path    points to txt under model_data, which is related to the data set you trained
    #                   Be sure to modify the classes_path to correspond to your own data set before training
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/insulator_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   The pre-training weight of the model
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolox_s.pth'
    #------------------------------------------------------#
    #   input_shape     The size of the input image, must be a multiple of 32.
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #------------------------------------------------------#
    #   The version of YOLOX used. This paper use YOLOX-s.
    #------------------------------------------------------#
    phi             = 's'
    #------------------------------------------------------------------#
    # mosaic Mosaic data enhancement.
    # mosaic_prob What is the probability of each step being enhanced with mosaic data, 50% by default.
    # mixup Whether to use mixup data enhancement, only valid if mosaic=True.
    # will only mixup mosaic enhanced images.
    # mixup_prob What is the probability of being enhanced with mixup data after mosaic, 50% by default.
    # Total mixup probability is mosaic_prob * mixup_prob.
    # special_aug_ratio refers to YoloX, because the training pictures generated by Mosaic are far from the real distribution of natural pictures.
    # When mosaic=True, this code will turn on mosaic in the special_aug_ratio range.
    # The default is the first 70% epoch, 100 generations will open 70 generations.
    # Cosine annealing algorithm parameters are set in lr_decay_type below
    #------------------------------------------------------------------#
    mosaic              = False
    mosaic_prob         = 1
    mixup               = False
    mixup_prob          = 1
    special_aug_ratio   = 0.75

    #----------------------------------------------------------------------------------------------------------------------------#
    # Init_Epoch: The epoch that model begins to train
    # Freeze_Epoch: Freeze training Epoch
    # (invalid when Freeze_Train=False)
    # Freeze_batch_size: The batch size of the model during freeze training
    # (invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 16
    #------------------------------------------------------------------#
    # UnFreeze_Epoch: Total epoch trained by the model
    # Unfreeze_batch_size: The batch size of the model during unfreeze training
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 50
    Unfreeze_batch_size = 8
    #------------------------------------------------------------------#
    # Freeze_Train: Whether to perform freeze training
    # Freeze trunk training before defrost training by default.
    #------------------------------------------------------------------#
    Freeze_Train        = False
    
    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decline
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         The maximum learning rate of the model
    #   Min_lr          The minimum learning rate of the model defaults to 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 5e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    # optimizer_type: Optimizer type to be used. The value can be adam or sgd
    # momentum: The momentum parameter used inside the optimizer
    # weight_decay weight attenuation prevents overfitting
    # adam results in a weight_decay error and is recommended to be set to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   The learning rate decay methods, include step and cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     How many epochs to save a weight
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        The weights and log files are saved in the folder
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    # eval_flag: Whether to evaluate during training. The evaluation object is the verification set
    # evaluate after eval_period epochs training
    # The mAP obtained here is the mAP of the verification set.
    #------------------------------------------------------------------#
    eval_flag           = False
    eval_period         = 5
    #------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multithreading to read data
    #------------------------------------------------------------------#
    num_workers         = 4

    #----------------------------------------------------#
    #   Get the image path and label
    #----------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #------------------------------------------------------#
    #   Set the graphics card to be used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0
        
    #----------------------------------------------------#
    #   Get classes and anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    #   Create a yolo model
    #------------------------------------------------------#
    model = YoloBody(num_classes, phi)
    weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mTips: It is normal that the head part does not load, and it is wrong that the Backbone part does not load.\033[0m")

    #----------------------#
    #   obtain loss function
    #----------------------#
    yolo_loss    = YOLOLoss(num_classes, fp16)
    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    # torch 1.2 does not support amp, we recommend using torch 1.7.1 and above to properly use fp16
    # So torch1.2 says "could not be resolved"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multiple graphics cards synchronize Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multiple graphics cards parallel running
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #----------------------------#
    #   Weight smoothing
    #----------------------------#
    ema = ModelEMA(model_train)
    
    #---------------------------#
    #   Read the txt corresponding to the data set
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
     
    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The data set is too small for training. Please expand the data set.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size above %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total training data amount of this run is %d, the Unfreeze_batch_size is %d, a total of %d epochs are trained, and the total training step size is %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training step size is %d, which is less than the recommended total step size %d, it is recommended to set the total generation to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    # Backbone feature extraction network features are universal, freezing training can speed up training
    # Also prevents weights from being destroyed at the beginning of training.
    # Init_Epoch is the start generation
    # Freeze_Epoch is the frozen training generation
    # UnFreeze_Epoch Total training generation
    # Prompt OOM or insufficient video memory please adjust small Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze training
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   If you do not want freeze training, set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 1e-4
        Init_lr_fit     = min(max(Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Select the optimizer based on optimizer_type
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        #   Obtain the formula of learning rate decay
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The data set is too small to continue training. Please expand the data set.")
        
        if ema:
            ema.updates     = epoch_step * Init_Epoch

        #---------------------------------------#
        #   Build the dataset loader.
        #---------------------------------------#
        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

        #----------------------#
        #   Record the eval map curve
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            # If the model has a freeze training part
            # Then unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 1e-4
                Init_lr_fit     = min(max(Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Obtain the formula of learning rate decay
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The data set is too small to continue training. Please expand the data set.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node
                    
                if ema:
                    ema.updates     = epoch_step * epoch
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
                        
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
