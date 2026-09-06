import torch
from torch.optim import AdamW
import random
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import time
import os

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrainLoop:
    def __init__(self, args, writer, model, diffusion, data, test_data, val_data, device):
        self.args = args
        self.writer = writer
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.test_data = test_data
        self.val_data = val_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=self.weight_decay)
        self.log_interval = args.log_interval
        self.best_rmse_random = 1e9
        self.warmup_steps=5
        self.min_lr = args.min_lr
        self.best_rmse = 1e9
        self.early_stop = 0
        
        self.mask_list = {'random_masking':[0.25, 0.5],'generation_masking':[1],'short_long_temporal_masking':[0.25,0.75]}
        # self.mask_list = {'random_masking':[0.375],'short_long_temporal_masking':[0.375,0.5]}


    def run_step(self, batch, step, index, mask_stg, mask_rate, name):
        self.opt.zero_grad()
        loss, num = self.forward_backward(batch, step, index=index, mask_stg = mask_stg, mask_rate = mask_rate, use_prior=self.args.use_prior, name = name)

        self._anneal_lr()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
        self.opt.step()
        return loss, num

    def Sample(self, test_data, step, mask_stg, mask_rate, seed=None, dataset='', use_prior=True, index=0, Type='val'):
        
        with torch.no_grad():
            error_mae, error_norm, error, num = 0.0, 0.0, 0.0, 0.0

            for name, batch in test_data:
                
                loss= self.model_forward(batch, self.model, mask_stg, mask_rate, dataset, use_prior, seed=seed)
                # error_norm += loss.item()
                error_norm += sum(loss['loss'])
                #
                num += loss['loss'].shape[0]
                # num2 += (1-mask).sum().item()

        loss_test = error_norm / num

        return loss_test
    
    def Evaluation(self, test_data, epoch, seed=None, best=True, use_prior=True, Type='val'):
        
        loss_list = []

        rmse_list = []
        rmse_key_result = {}

        for index, dataset_name in enumerate(self.args.dataset.split('_')):

            rmse_key_result[dataset_name] = {}

            if self.args.mask_strategy_random != 'none':
                for s in self.mask_list:
                    for m in self.mask_list[s]:
                        result= self.Sample(test_data, epoch, mask_stg=s, mask_rate= m, seed=seed, dataset=dataset_name, use_prior=use_prior, index=index, Type=Type)
                        rmse_list.append(result)
                        # loss_list.append(loss_test)
                        if s not in rmse_key_result[dataset_name]:
                            rmse_key_result[dataset_name][s] = {}
                        rmse_key_result[dataset_name][s][m] = result
                        
                        if Type == 'val':
                            self.writer.add_scalar('Evaluation/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                        elif Type == 'test':
                            self.writer.add_scalar('Test_RMSE/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                            # self.writer.add_scalar('Test_MAE/MAE-{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), mae, epoch)

            else:
                s = self.args.mask_strategy
                m = self.args.mask_ratio
                result = self.Sample(test_data, epoch, mask_stg=s, mask_rate=m, seed=seed, dataset = dataset_name, use_prior=use_prior, index=index, Type=Type)
                rmse_list.append(result)
                # loss_list.append(loss_test)
                if s not in rmse_key_result[dataset_name]:
                    rmse_key_result[dataset_name][s] = {}
                rmse_key_result[dataset_name][s][m] = {'rmse':result}
                
                if Type == 'val':
                    self.writer.add_scalar('Evaluation/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
                    print('look',result)
                elif Type == 'test':
                    self.writer.add_scalar('Test_RMSE/Stage-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), result, epoch)
                    # self.writer.add_scalar('Test_MAE/Stage-MAE-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), mae, epoch)
                    
        
        # loss_test = np.mean(rmse_list)
        loss_test = np.mean(np.array([tensor.cpu().numpy() for tensor in rmse_list]))

        if best:
            is_break = self.best_model_save(epoch, loss_test, rmse_key_result)
            return is_break

        else:
            return  loss_test, rmse_key_result

    def best_model_save(self, step, rmse, rmse_key_result):
        if rmse < self.best_rmse:
            self.early_stop = 0
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_' + self.args.ts_state + '_' + self.args.datatype + '.pkl')
            self.best_rmse = rmse
            self.writer.add_scalar('Evaluation/RMSE_best', self.best_rmse, step)
            print('\nRMSE_best:{}\n'.format(self.best_rmse))
            print(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result.txt', 'w') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            return 'save'

        elif self.args.stage in [0,1,2]:
            self.early_stop += 1
            print('\nRMSE:{}, RMSE_best:{}, early_stop:{}\n'.format(rmse, self.best_rmse, self.early_stop))
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('RMSE:{}, not optimized, early_stop:{}\n'.format(rmse, self.early_stop))
            if self.early_stop >= self.args.early_stop:
                print('Early stop!')
                print('From best_model_save:')
                self.evaluating()

                with open(self.args.model_path+'result.txt', 'a') as f:
                    f.write('Early stop!\n')
                with open(self.args.model_path+'result_all.txt', 'a') as f:
                    f.write('Early stop!\n')
                if self.args.stage in [0,2]:
                    exit()
                elif self.args.stage == 1:
                    return 'break_1_stage'

            return 'none'
        
    def mask_select(self):
        if self.args.mask_strategy_random == 'none': # 'none' or 'batch'
            mask_strategy = self.args.mask_strategy
            mask_ratio = self.args.mask_ratio
        else:
            # mask_strategy=random.choice(['short_long_temporal_masking'])
            mask_strategy=random.choice(['random_masking','short_long_temporal_masking','generation_masking'])

            mask_ratio=random.choice(self.mask_list[mask_strategy])

        return mask_strategy, mask_ratio

    def run_loop(self, args, loss_to_save):
        step = 0
        
        self.Evaluation(self.val_data, 0, best=True, use_prior=args.use_prior, Type='val')
        for epoch in range(self.args.total_epoches):
            print('Training')

            self.step = epoch
            
            loss_all, num_all = 0.0, 0.0
            start = time.time()
            for name, batch in self.data:
                mask_strategy, mask_ratio = self.mask_select()
                loss, num = self.run_step(batch, step,index=0, mask_stg=mask_strategy, mask_rate=mask_ratio, name = name)
                step += 1
                loss_all += loss * num
                # loss_real_all += loss_real * num
                num_all += num
                # num_all2 += num2

            end = time.time()
            print('training time:{} min'.format(round((end-start)/60.0,2)))
            print('epoch:{}, training loss:{}'.format(epoch, loss_all / num_all))

            loss_to_save.append(loss_all / num_all)

            if epoch >= 10 or self.args.mode!='training':
                self.writer.add_scalar('Training/Stage_{}_Loss_epoch'.format(self.args.stage), loss_all / num_all, epoch)


            if epoch % self.log_interval == 0 and epoch > 0 or epoch == 10 or epoch == self.args.total_epoches-1:
                print('Evaluation')
                #X = [lst for _, lst in self.data]
                #data_t = [X]
                is_break = self.Evaluation(self.val_data, epoch, best=True, use_prior=args.use_prior, Type='val')

                if is_break == 'break_1_stage':
                    break

                if is_break == 'save':
                    print('test evaluate!')

                    rmse_test, rmse_key_test = self.Evaluation(self.test_data, epoch, best=False, use_prior=args.use_prior, Type='test')
                    print('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                    print(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result_all.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')
        
        self.evaluating()

        if self.args.stage == 1 and epoch<self.args.total_epoches-1:
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('\nnot enough epoch for stage 1!\n')
        elif self.args.stage == 2 and epoch<self.args.total_epoches-1:
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('\nnot enough epoch for stage 2!\n')
        elif self.args.stage == 0 and epoch<self.args.total_epoches-1:
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('\nnot enough epoch!\n')
        
        return loss_to_save

    def evaluating(self):
        with torch.no_grad():
            print('Generate samples:')
            self.model.eval()
            error_before_scaler, error_after_scaler =0.0, 0.0
            target_to_save = []
            sample_to_save = []
            mask_to_save = []
            cond_to_save = []
            name_to_save = []
            for index_t, dataset_name_t in enumerate(self.args.dataset.split('_')):

                for index_mask, (name, batch2) in enumerate(self.test_data):
                    model_kwargs_t = None
                    x_start = batch2[0].cuda() if self.args.ts_state == 'student' else batch2[1].cuda()
                    x_calc = batch2[1].cuda().unsqueeze(1)
                    cond = batch2[2].cuda()
                    mag = batch2[4].cuda() if self.args.use_mag == True else None
                    traj = batch2[5].cuda() if self.args.use_mag == True else None
                    if self.args.use_prior:
                        ref_emb = batch2[6].cuda() if self.args.use_mag == True else batch2[4].cuda()
                    else:
                        ref_emb = None

                    cond_tuple = (cond, mag, traj, ref_emb)

                    mask_strategy, mask_ratio = self.mask_select()

                    if self.args.task == 'mix':
                        if index_mask % 3 == 0:
                            mask_strategy = 'short_long_temporal_masking'
                            mask_ratio = 0.25
                        elif index_mask % 3 == 1:
                            mask_strategy = 'short_long_temporal_masking'
                            mask_ratio = 0.75
                        else:
                            mask_strategy = 'generation_masking'
                            mask_ratio = 1
                    elif self.args.task == 'generation':
                        mask_strategy = 'generation_masking'
                        mask_ratio = 1
                    elif self.args.task == 'prediciton':
                        if index_mask % 2 == 0:
                            mask_strategy = 'short_long_temporal_masking'
                            mask_ratio = 0.25
                        elif index_mask % 2 == 1:
                            mask_strategy = 'short_long_temporal_masking'
                            mask_ratio = 0.75
                    elif self.args.task == 'long_prediction':
                        mask_strategy = 'short_long_temporal_masking'
                        mask_ratio = 0.75
                    elif self.args.task == 'short_prediction':
                        mask_strategy = 'short_long_temporal_masking'
                        mask_ratio = 0.25

                    mask_origin = self.function_dict[mask_strategy](self, x_start, mask_ratio=mask_ratio)
                    x_start = x_start.unsqueeze(1)
                    x_start_masked = mask_origin * x_start

                    sample, mask = self.diffusion.p_sample_loop(
                        self.model, x_start.shape, x_start, x_calc, cond_tuple, mask_origin, x_start_masked, datatype=name, clip_denoised=True, model_kwargs=model_kwargs_t, progress=True,
                        device=self.device
                    )
                    target = batch2[3].unsqueeze(1) # 未经scaler的原始数据
                    samples = sample * mask.cuda() + target.cuda() * (1-mask)

                    error_before_scaler += mean_squared_error(samples.reshape(-1,1).detach().cpu().numpy(),self.args.scaler[dataset_name_t].transform(target.reshape(-1,1).detach().cpu().numpy()))/batch2[0].shape[0]
                    error_after_scaler += mean_squared_error(self.args.scaler[dataset_name_t].inverse_transform(samples.reshape(-1,1).detach().cpu().numpy()),target.reshape(-1,1).detach().cpu().numpy())/batch2[0].shape[0]

                    save_tar = target.detach().cpu().numpy()
                    save_gen = samples.detach().cpu().numpy()
                    save_cond = cond.detach().cpu().numpy()

                    target_to_save.append(save_tar)
                    sample_to_save.append(save_gen)
                    mask_to_save.append(mask_origin.squeeze(1).detach().cpu().numpy())
                    cond_to_save.append(save_cond)
                    name_to_save.append(name)
            print('error_before_scaler:', error_before_scaler)
            print('error_after_scaler:', error_after_scaler)

            filename_gen = f'generate.npz'
            filename_tar = f'target.npz'
            filename_mask = f'mask.npz'
            filename_cond = f'cond.npz'
            filename_type = f'datatype.npz'

            save_dir = self.args.model_path + f'data_save_{self.args.ts_state}/'
            os.makedirs(save_dir, exist_ok=True)

            np.savez(save_dir + filename_gen, gen_traffic=sample_to_save)
            np.savez(save_dir + filename_tar, tar_traffic=target_to_save)
            np.savez(save_dir + filename_mask, mask=mask_to_save)
            np.savez(save_dir + filename_cond, cond=cond_to_save)
            np.savez(save_dir + filename_type, datatype=name_to_save)


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # N, L, D = x.shape  # batch, length, dim
        
        B, T = x.shape  # batch, length,
        x = x.reshape(B,-1,self.args.t_patch_size)
        B, C, _ = x.shape
        num_elements = C
        num_ones = int(num_elements * mask_ratio)

        mask = torch.zeros_like(x.squeeze(1), dtype=torch.bool)
        for b in range(B):
            # Create a flattened array of indices and shuffle it
            indices = torch.randperm(num_elements, device=x.device)
            ones_indices = indices[:num_ones]
            for j in ones_indices:
                mask[b,j,:]= 1
        mask = mask.reshape(B,1,T)
        return  mask.float()

    def small_tube_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # N, L, D = x.shape  # batch, length, dim

        B, T= x.shape
        # t = T//self.args.t_patch_size
        # x = x.reshape(B,1,t,self.args.t_patch_size)

        mask = torch.ones_like(x, dtype=torch.float32, device=x.device)
        mask = mask.reshape(B, 1, T)

        return mask.float()

    def short_long_temporal_masking(self, x, mask_ratio):
        """
        根据 mask_ratio大小控制短时间mask和长时间mask
        """
        
        B, T= x.shape
        t = T//self.args.t_patch_size

        x = x.reshape(B,1,t,self.args.t_patch_size)

        mask = torch.zeros_like(x, dtype=torch.float32, device=x.device)

        num_times_to_mask = int(t * mask_ratio)  # 计算需要mask的时间步数
        start_time = t - num_times_to_mask  # 计算时间维度的mask开始点

        # 为所有空间位置的最后T*m个时间步设置mask
        mask[:, :, start_time:, :] = 1
        mask = mask.reshape(B,1,T)
        return mask.float()

    function_dict = {
        'random_masking': random_masking,
        'generation_masking': small_tube_masking,
        'short_long_temporal_masking': short_long_temporal_masking
    }

    def model_forward(self, batch, model, mask_stg, mask_rate, datatype, use_prior, seed = None):

        batch = [i.to(self.device) for i in batch]
        t = torch.randint(0, self.diffusion.num_timesteps, (batch[0].shape[0],), device=self.device)
        x_start = batch[0] if self.args.ts_state == 'student' else batch[1]
        x_start_calc = batch[1]
        model_kwargs = None
        cond = batch[2]
        mag = batch[4] if self.args.use_mag == True else None
        traj = batch[5] if self.args.use_mag == True else None
        if self.args.use_prior:
            ref_emb = batch[6].cuda() if self.args.use_mag == True else batch[4].cuda()
        else:
            ref_emb = None

        cond_tuple = (cond, mag, traj, ref_emb)
        
        mask_origin = self.function_dict[mask_stg](self,x_start, mask_ratio = mask_rate)
        # x_start_masked = mask_origin * mask_origin
        loss= self.diffusion.training_losses(model, x_start.unsqueeze(1), x_start_calc.unsqueeze(1), cond_tuple, mask_origin, t, datatype, use_prior, model_kwargs)

        return loss

    def forward_backward(self, batch, step, index, mask_stg, mask_rate, use_prior, name=None):

        loss_multi= self.model_forward(batch, self.model,  mask_stg=mask_stg, mask_rate=mask_rate, datatype=name, use_prior=use_prior)

        num = loss_multi['loss'].shape[0]
        loss = sum(loss_multi['loss'])/num
        loss.backward()

        self.writer.add_scalar('Training/Loss_step', np.sqrt(loss.detach().cpu().numpy()), step)
        return loss.item(), num

    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step+1) / self.warmup_steps
        elif self.step < self.lr_anneal_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (self.step - self.warmup_steps)
                    / (self.lr_anneal_steps - self.warmup_steps)
                )
            )
        else:
            lr = self.min_lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        self.writer.add_scalar('Training/LR', lr, self.step)
        return lr