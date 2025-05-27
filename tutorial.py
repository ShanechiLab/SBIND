import os
import numpy as np
import torch
import copy
import sbind.sbind as sbind
import sbind.sbind_trainer as sbind_trainer
from sbind.datasets.timeseries_dataset import create_dataloader_from_data
import torch.optim.lr_scheduler as lr_scheduler
from sbind.models.model_helpers import create_general_model_configs

from sbind.utility.utils import DataSplitter, eval_prediction


def main():
    # Placeholder data
    var_y = np.random.random((1000, 1, 128, 128))
    var_z = np.random.random((1000, 14))
    num_z = var_z.shape[1]
    temporal_mask = np.ones(var_y.shape[0], dtype=np.int32)
    data_splitter = DataSplitter(y=var_y, z=var_z.astype('int32'), temp_mask=temporal_mask,
                                 val_train_ratio=0.2)

    model_config = {
        "indep_msa": False,
        "use_lstm": False,
        "stateful": True,
        "unified_K": True,

        "Cy_observation_model": "image_gaussian",
        "step_ahead_prediction": None
    }
    trainer_args = {
        "optim_kwargs": {
            "lr": 1e-3,
            "weight_decay": 1e-7
        },
        "verbose": 1,
        "clip_gradients": 0.5
    }
    fit_args = {
        "start_from_epoch": 5,
        "early_stopping_patience": 5,
        "early_stopping_measure": "val_loss"
    }

    epochs = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    folds = 5

    metrics = ["CC", "MSE", "R2", "NRMSE", "MAE", "SSIM"]
    behavior_metrics = ["CC", "MSE", "R2", "NRMSE"]

    ifold = 1
    split_data = data_splitter.split_data_for_fold(folds=folds, ifold=ifold)
    y_train, y_val, y_test = split_data['yTrain'], split_data['yVal'], split_data['yTest']
    z_train, z_val, z_test = split_data['zTrain'], split_data['zVal'], split_data['zTest']
    mask_train, mask_val, mask_test = split_data['maskTrain'], split_data['maskVal'], split_data['maskTest']
    seq_len = 63
    batch_size = 7

    dataloader = create_dataloader_from_data(
        y_train, z_train, mask_train, seq_len=seq_len, batch_size=batch_size, shuffle=False,
        stateful=model_config["stateful"], step_ahead=0, num_workers=8)
    dataloader_val = create_dataloader_from_data(
        y_val, z_val, mask_val, seq_len=seq_len, batch_size=batch_size, shuffle=False,
        stateful=model_config["stateful"],
        step_ahead=0, num_workers=8)
    dataloader_test = create_dataloader_from_data(
        y_test, z_test, mask_test, seq_len=seq_len, batch_size=batch_size, shuffle=False,
        stateful=model_config["stateful"],
        step_ahead=0, num_workers=8)

    patch_size = 8
    num_heads = 8
    embedding_dim = 256
    pos_embedding = "learnable_1d"
    att_reduced_ch = 8
    num_x = 48
    ds = 2
    kernel_size = 3

    K_strides = [2, 2, 1]
    Cy_strides = [2, 2, 1]
    K_use_residuals = [False, True, False]
    Cy_use_residuals = [False, True, False]
    K_use_pixelshuffle = [None, None, None]
    Cy_use_pixelshuffle = [False, False, False]
    K_encoder_layer_type = ["conv", "conv", "conv"]
    Cy_encoder_layer_type = ["conv", "conv", "conv"]

    K_kernel_size = 5
    Cy_kernel_size = 5
    K_channel_size = 32
    Cy_channel_size = 32

    Cz_kernel_size = [5, 5, 5, 5]
    Cz_strides = [2, 2, 2, 2]
    Cz_use_residuals = [None, None, None, None, None]
    Cz_use_pixelshuffle = [None, None, None, None, None]
    Cz_encoder_layer_type = ["conv", "conv", "conv", "conv", "fcn"]
    Cz_dropout = [0.4, 0.4, 0.4, 0.4, 0.0]
    Cz_channel_size = 64
    Cz_batch_norm = True
    Cz_activation = "leaky_relu"
    y_train_shape = y_train.shape

    A_config, K_config, Cy_config, Cz_config, _, _, _ = create_general_model_configs(
        patch_size, num_heads, embedding_dim, pos_embedding, num_x, num_z, y_train_shape, ds,
        kernel_size, K_kernel_size, Cy_kernel_size, Cz_kernel_size, K_channel_size,
        Cy_channel_size, Cz_channel_size,
        K_strides, Cy_strides, Cz_strides, K_use_residuals, Cy_use_residuals, Cz_use_residuals,
        K_use_pixelshuffle,
        Cy_use_pixelshuffle, Cz_use_pixelshuffle,
        K_encoder_layer_type, Cy_encoder_layer_type, Cz_encoder_layer_type, Cz_dropout,
        Cz_batch_norm, Cz_activation, True, att_reduced_ch)
    num_stage1 = 8

    model_config.update({
        'A_config': copy.deepcopy(A_config),
        'K_config': copy.deepcopy(K_config),
        'Cy_config': copy.deepcopy(Cy_config),
        'nx': num_x,
        'Cz_config': copy.deepcopy(Cz_config),
        'n1': num_stage1,
        "device": device,
        "fit_Cz2": False,
    })
    model = sbind.SBIND(**model_config).to(device)
    print(model)
    trainer_args.update({
        "lambda_l1": 2.0,
        "lambda_grad": 0.3,
    })

    fit_args.update({
        "device": device,
        "val_every_epoch": 1,
        "check_point_args": {
            'checkpoint_every_epoch': 25,
            'name': 'MODEL_FOLDER',  # Changed name to a generic name
            'base_path': os.path.dirname(os.path.abspath(__file__)),
        },
        "scheduler_type": lr_scheduler.StepLR,
        "step_size": 700,
        "gamma": 0.4
    })

    stage1, stage2 = (num_stage1 > 0), (num_x - num_stage1 > 0)

    model_trainer = sbind_trainer.SBINDTrainer(model, stage1, stage2, **trainer_args)
    history = model_trainer.fit(epochs, dataloader, dataloader_val, **fit_args)

    y_train_pred, x_train_pred, z_train_pred = model_trainer.predict(dataloader, device=device)
    y_val_pred, x_val_pred, z_val_pred = model_trainer.predict(dataloader_val, device=device)
    y_test_pred, x_test_pred, z_test_pred = model_trainer.predict(dataloader_test, device=device)

    testPerf = eval_prediction(y_train[:y_train_pred.shape[0]], y_train_pred, behavior_metrics[0])


if __name__ == "__main__":
    main()