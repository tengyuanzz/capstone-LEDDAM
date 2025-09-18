from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
import matplotlib
matplotlib.use('inline')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE' or self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'MAE' or self.args.loss == 'mae':
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            preds=[]
            trues=[]
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                # print('vali:',i)
                batch_x = batch_x.float().to(self.device,non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:,:].float()
                # print('batch_x', batch_x[:1])
                # encoder - decoder
                if self.args.use_amp:
                    # print('outputs1:')
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    # print('outputs2:')
                    outputs = self.model(batch_x)
                    # print(outputs[:1])
                # print('outputs3:', outputs[:1])
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().numpy()
                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)
        else:
            raise IndexError("Preds contain nothing")
        mse,mae= metric(preds, trues)
        vali_loss=mae if criterion == 'MAE' or criterion == 'mae' else mse
        self.model.train()
        torch.cuda.empty_cache()
        return vali_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device,non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:,:].float().to(self.device,non_blocking=True)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss= self.vali(vali_data, vali_loader, self.args.loss)
            test_loss = self.vali(test_data, test_loader, self.args.loss)

            print("Epoch: {}, Steps: {} | Train Loss: {:.3f}  vali_loss: {:.3f}   test_loss: {:.3f} ".format(epoch + 1, train_steps, train_loss,  vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        torch.cuda.empty_cache()

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        # if os.path.exists(os.path.join(os.path.join(path, 'checkpoint.pth'))):
        #     os.remove(os.path.join(os.path.join(path, 'checkpoint.pth')))
        #     print('Model weights deleted.')

        head = f'./test_dict/{self.args.data_path[:-4]}/'
        
        tail= f'{self.args.model}/{self.args.loss}/bz_{self.args.batch_size}/lr_{self.args.learning_rate}/'
        
        dict_path= head+tail
        
        
        if not os.path.exists(dict_path):
                os.makedirs(dict_path)

        self.model.eval()
        with torch.no_grad():
            preds=[]
            trues=[]
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device,non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:,:].float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().numpy()

                pred = outputs
                true = batch_y
                
                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)
        else:
            preds=preds[0]
            trues=trues[0]
        print('test shape:', preds.shape, trues.shape)

        # --- Inverse Scaling ---
        preds_orig_scale = preds
        trues_orig_scale = trues
        if hasattr(test_data, 'inverse_transform') and hasattr(test_data, 'scaler'):
            print('Attempting inverse transform...')
            try:
                # Use the inverse_transform method from the specific dataset object
                preds_orig_scale = test_data.inverse_transform(torch.from_numpy(preds)).numpy()
                trues_orig_scale = test_data.inverse_transform(torch.from_numpy(trues)).numpy()
                print('Shapes after inverse transform:', preds_orig_scale.shape, trues_orig_scale.shape)
            except Exception as e:
                print(f"WARNING: Inverse transform failed: {e}. Using scaled data for metrics/plotting.")
                # Fallback to using potentially scaled data
                preds_orig_scale = preds
                trues_orig_scale = trues
        # ------------------------

        mse, mae = metric(preds_orig_scale, trues_orig_scale)
        print('Metrics (Original Scale): mse:  {:.3f}  mae:  {:.3f}'.format(mse, mae))
        my_dict = {
            'mse': "{:.3f}".format(mse),
            'mae': "{:.3f}".format(mae),
        }
        metrics_filepath = os.path.join(dict_path, 'records.json')
        with open(metrics_filepath, 'w') as f:
            json.dump(my_dict, f) # Keep original dump format (no indent)

        pred_save_path = os.path.join(dict_path, 'pred.npy')
        true_save_path = os.path.join(dict_path, 'true.npy')
        np.save(pred_save_path, preds_orig_scale)
        np.save(true_save_path, trues_orig_scale)

        try:
            # Ensure there is data to plot
            if preds.shape[0] > 0:
                # Plot the first sample's prediction horizon vs true values
                # Assumes target dimension C is the last one, and we plot the first target C=0
                sample_idx = 0 # Plot the first sample generated by the sliding window
                target_idx = 0 # Plot the first target variable

                fig = plt.figure(figsize=(15, 6)) # Store figure object
                plt.plot(trues[sample_idx, :, target_idx], label='GroundTruth (Scaled)')
                plt.plot(preds[sample_idx, :, target_idx], label='Prediction (Scaled)')
                plt.title(f'Prediction vs Ground Truth (Sample {sample_idx}, Target {target_idx}) - Scaled Data')
                plt.xlabel(f'Time Step in Prediction Horizon (Total {self.args.pred_len})')
                plt.ylabel('Scaled Value')
                plt.legend()
                plot_save_path = os.path.join(dict_path, f'pred_vs_true_sample{sample_idx}_scaled.png')
                plt.savefig(plot_save_path)
                plt.show()
                print(f'Plot saved to {plot_save_path}')
                plt.close(fig) # Close the specific figure
            else:
                 print("No samples available to plot.")
        except Exception as e:
            print(f"An error occurred during plotting: {e}")

        torch.cuda.empty_cache()
        return