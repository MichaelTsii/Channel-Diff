import argparse
import random
import os
from models_with_mask_scale import DiT_models
from train import TrainLoop
import setproctitle
import torch
from DataLoader import data_load
from utils import *
import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from diffusion import create_diffusion
import datetime
import re

# 数据集基本参数
dataset_list = 'RSRP-image-3' # RSRP-CPGMCM, RSRP-image-96
model_size = 'DiT-S/8'
time_length = 64
gamma = 0

task = 'generation' # 'mix', 'prediction'
prompt_state = 'test' #'load','train','test','stu-train', 'zs', 'fs'
save_folder = 'RSRP-image-3-t50-s320_P_1204_150625' # 仅当 prompt_state == 'test' 时有效
fs_rate = 0.1

use_BH_map = True
use_AL_map = True
visual_model = 'DETR' # CNN, MLP, DETR
use_Hessian = True

use_prior = True
# ray_condition = 'Nb'
ray_condition = 'OF' # 新定义的
use_ref_emb = True

teacher_epoches = 50
student_only = True if teacher_epoches == 0 else False
student_epoches = 320

os.environ["CUDA_VISIBLE_DEVICES"] = '4' # cuda
process_name = dataset_list + "@qxq"

def setup_init(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # th.manual_seed(seed)
    # th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    # th.backends.cudnn.deterministic = True

def dev(device_id):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")

def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        task = 'short',
        use_prior=use_prior,
        early_stop = 12,
        weight_decay=1e-4,
        log_interval=20,

        # 基本参数 ——————————————
        batch_size = 48,
        teacher_epoches = teacher_epoches,
        student_epoches = student_epoches,

        device_id='0', # 这里不要改变
        machine = 'machine_name',
        mask_ratio = 1,
        lr_anneal_steps = 300,
        patch_size = 1,
        random=True,
        t_patch_size = 1,
        size = 'small',
        clip_grad = 0.5,
        mask_strategy = 'generation_masking', # ['generation_masking', 'random_masking', 'short_long_temporal_masking'], # 'random'
        mask_strategy_random = 'none', # ['none','batch']

        # zero/few-shot 相关
        fs_rate = fs_rate,
        
        # 多属性图相关
        use_BH_map = use_BH_map,
        use_AL_map = use_AL_map,
        use_mag = use_BH_map or use_AL_map,
        map_dim = use_AL_map + use_BH_map,
        visual_model = visual_model,
        use_Hessian = use_Hessian,
        ray_condition = ray_condition,
        use_ref_emb = use_ref_emb,
        submag_length = 17,

        mode='training',
        file_load_path = '',
        min_lr = 1e-5,
        dataset = dataset_list,
        stage = 0,
        no_qkv_bias = 0,
        pos_emb = 'SinCos',
        used_data = '',
        process_name = process_name,
        gamma = gamma,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    th.autograd.set_detect_anomaly(True)

    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-t{}-s{}@qxq".format(args.dataset, args.teacher_epoches, args.student_epoches))
    setup_init(100)

    args.task = task
    args.prompt_state = prompt_state
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    args.folder = '{}/'.format(f"{dataset_list}-t{args.teacher_epoches}-s{args.student_epoches}_P_{current_time}") if args.use_prior==True \
        else '{}/'.format(f"{dataset_list}-t{args.teacher_epoches}-s{args.student_epoches}_{current_time}")
    args.datatype = dataset_list
    args.time_length = time_length
    args.model_path = './experiments/{}'.format(args.folder) 
    args.save_folder = save_folder
    print("Folder: ", args.folder)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')
    
    # 生成 readme.txt 文件
    with open(args.model_path + "set.txt", "w") as f:
        f.write("# 数据集基本参数\n")
        f.write(f"dataset_list: {dataset_list}\n")
        f.write(f"gamma: {gamma}\n\n")
        
        f.write("# 结构设置\n")
        f.write(f"use_BH_map: {use_BH_map}\n")
        f.write(f"use_AL_map: {use_AL_map}\n")
        f.write(f"visual_model: {visual_model}\n\n")
        
        f.write("# 训练参数\n")
        f.write(f"student_only: {student_only}\n")
        f.write(f"teacher_epoches: {teacher_epoches}\n")
        f.write(f"student_epoches: {student_epoches}\n")

    print('start data load')
    data, test_data, val_data, args.scaler = data_load(args)
    
    writer = SummaryWriter(log_dir=args.model_path, flush_secs=5)

    device = dev(args.device_id)
    args.device = device

    model = DiT_models[model_size](args=args).to(device)
    diffusion = create_diffusion(args, timestep_respacing="", diffusion_steps=400)

    loss_to_save = []

    # 如果直接读取已训练模型
    if prompt_state in ['test', 'zs']:
        args.ts_state = 'student'

        model_folder = './experiments/' + save_folder + '/model_save/'
        if prompt_state == 'test':
            file_path = model_folder + 'model_best_student_' + dataset_list + '.pkl'
        else:
            file_path = model_folder + 'model_best_teacher_' + re.split(r'[/_]', args.save_folder)[-3] + '.pkl'
        state_dict = torch.load(file_path)
        model.load_state_dict(state_dict)
        args.total_epoches = args.student_epoches

        TrainLoop(
            args=args,
            writer=writer,
            model=model,
            diffusion=diffusion,
            data=data,
            test_data=test_data, 
            val_data=val_data,
            device=device
        ).evaluating()
    
    elif prompt_state in ['stu-train', 'fs']:
        if args.student_epoches:

            model_folder = './experiments/' + save_folder + '/model_save/'
            if prompt_state == 'stu-train':
                file_path = model_folder + 'model_best_teacher_' + dataset_list + '.pkl'
            else:
                file_path = model_folder + 'model_best_teacher_' + re.split(r'[/_]', args.save_folder)[-3] + '.pkl'
           
            state_dict = torch.load(file_path)
            model.load_state_dict(state_dict)

            args.ts_state = 'student'
            args.total_epoches = args.student_epoches
            train_loop_real = TrainLoop(
                args=args,
                writer=writer,
                model=model,
                diffusion=diffusion,
                data=data,
                test_data=test_data, 
                val_data=val_data,
                device=device
            )
            loss_to_save = train_loop_real.run_loop(args, loss_to_save)
        
        np.save(args.model_path + '/model_save/loss_curve_student.npy', np.array(loss_to_save))
    
    else: # 两阶段训练
        if student_only == False:
            args.ts_state = 'teacher'
            args.total_epoches = args.teacher_epoches
            train_loop_calc = TrainLoop(
                args=args,
                writer=writer,
                model=model,
                diffusion=diffusion,
                data=data,
                test_data=test_data, 
                val_data=val_data,
                device=device
            )
            loss_to_save = train_loop_calc.run_loop(args, loss_to_save)
            model = train_loop_calc.model
        
        np.save(args.model_path + '/model_save/loss_curve_teacher.npy', np.array(loss_to_save))
        
        if args.student_epoches:
            args.ts_state = 'student'
            args.total_epoches = args.student_epoches
            train_loop_real = TrainLoop(
                args=args,
                writer=writer,
                model=model,
                diffusion=diffusion,
                data=data,
                test_data=test_data, 
                val_data=val_data,
                device=device
            )
            loss_to_save = train_loop_real.run_loop(args, loss_to_save)

        np.save(args.model_path + '/model_save/loss_curve_student.npy', np.array(loss_to_save))

if __name__ == "__main__":
    main()