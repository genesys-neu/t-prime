import os
import numpy as np
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
import sys
sys.path.append(proj_root_dir)
from preprocessing.TPrime_dataset import TPrimeDataset
from ray.air import session, Checkpoint
from typing import Dict
import torch
from torch import nn
from torch.utils.data import DataLoader
from ray.train.torch import TorchTrainer, TorchPredictor
from ray.air.config import ScalingConfig
import ray.train as train
import pickle
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import tqdm
from model_cnn1d import Baseline_CNN1D
from model_AMCNet import AMC_Net
from model_ResNet import ResNet
from model_LSTM import LSTM_ap
from model_MCFormer import MCformer
from confusion_matrix import plot_confmatrix
import wandb
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_epoch(dataloader, model, loss_fn, optimizer, use_ray=False):
    if use_ray:
        size = len(dataloader.dataset) // session.get_world_size()
    else:
        size = len(dataloader.dataset)
    model.train()
    correct = 0
    loss = 0
    for batch, (X, y) in tqdm(enumerate(dataloader), desc="Training epochs.."):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Cached files: ", len(dataloader.dataset.signal_cache.cache.keys()))
    correct /= size
    print(f"Train Error: \n "
          f"Accuracy: {(100 * correct):>0.1f}%, "
          )
    return loss, correct

from sklearn.metrics import confusion_matrix as conf_mat
def validate_epoch(dataloader, model, loss_fn, Nclasses, use_ray=False):
    if use_ray:
        size = len(dataloader.dataset) // session.get_world_size()
    else:
        size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    conf_matrix = np.zeros((Nclasses, Nclasses))
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_cpu = y.to('cpu')
            pred_cpu = pred.to('cpu')
            conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(Nclasses)))
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )

    return test_loss, correct, conf_matrix

def train_func(config: Dict):

    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    Nclass = config["Nclass"]
    use_ray = config['useRay']
    slice_len = config['slice_len']
    num_feats = config['num_feats']
    num_channels = config['num_chans']
    device = config['device']
    logdir = config['cp_path']
    is_wandb = config['is_wandb']
    os.makedirs(logdir, exist_ok=True)

    if not use_ray:
        worker_batch_size = batch_size
    else:
        worker_batch_size = batch_size // session.get_world_size()

    # Create data loaders.
    train_dataloader = DataLoader(ds_train, batch_size=worker_batch_size, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=worker_batch_size, shuffle=True)

    if use_ray:
        train_dataloader = train.torch.prepare_data_loader(train_dataloader)
        test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    # Create model.
    if config['pytorch_model'] == 'baseline_cnn1d':
        model = Baseline_CNN1D(classes=Nclass, numChannels=num_channels, slice_len=slice_len, normalize=config['normalize'])
        loss_fn = nn.NLLLoss()
    elif config['pytorch_model'] == 'AMCNet':
        model = AMC_Net(num_classes=Nclass, sig_len=slice_len, extend_channel=36, latent_dim=512, num_heads=2, conv_chan_list=None)
        loss_fn = nn.CrossEntropyLoss()
    elif config['pytorch_model'] == 'ResNet':
        model = ResNet(num_classes=Nclass, num_samples=slice_len, iq_dim=2, kernel_size=3, pool_size=2)
        loss_fn = nn.CrossEntropyLoss()
    elif config['pytorch_model'] == 'LSTM':
        model = LSTM_ap(input_shape=[slice_len, 2], hidden_size=128, output_size=Nclass, num_layers=2)
        loss_fn = nn.CrossEntropyLoss()
    elif config['pytorch_model'] == 'MCformer':
        model = MCformer(classes=Nclass, hidden_size=32, kernel_size=65, num_heads=4)
        loss_fn = nn.CrossEntropyLoss()

    print("Model:", config['pytorch_model'], "Num. Parameters:", count_parameters(model) )


    if use_ray:   
        model = train.torch.prepare_model(model)
    else:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
    loss_results = []
    best_loss = np.inf
    if is_wandb:
        wandb.watch(model, log_freq=10)
    normalize_flag = 'norm.' if config['normalize'] else ''
    for e in range(epochs):
        print(f"Epoch {e + 1}\n-------------------------------")
        if is_wandb:
            wandb.log({'Epoch': e}, step=e)
        tr_loss, tr_acc = train_epoch(train_dataloader, model, loss_fn, optimizer, use_ray)
        if is_wandb:
            wandb.log({'Tr_loss': tr_loss}, step=e)
            wandb.log({'Tr_acc': tr_acc}, step=e)
        loss, acc, conf_matrix = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass, use_ray=use_ray)
        if is_wandb:
            wandb.log({'Val_loss': loss}, step=e)
            wandb.log({'Val_acc': acc}, step=e)
        scheduler.step(loss)
        loss_results.append(loss)
        if use_ray:
            if best_loss > loss:
                best_loss = loss

            # store checkpoint only if the loss has improved
            state_dict = model.state_dict()
            consume_prefix_in_state_dict_if_present(state_dict, "module.")
            checkpoint = Checkpoint.from_dict(
                dict(epoch=e, model_weights=state_dict)
            )

            session.report(dict(loss=loss), checkpoint=checkpoint)
        else:
            pkl_file = 'conf_matrix.best.pkl'
            if best_loss > loss:
                best_loss = loss
                pickle.dump(conf_matrix, open(os.path.join(logdir, pkl_file), 'wb'))
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(logdir,f'model.{config["pytorch_model"]}.{config["wchannel"]}.{normalize_flag}{config["snr_dB"]}.pt'))
                df_confmat = plot_confmatrix(logdir, pkl_file, train_config['class_labels'], 'conf_mat_epoch'+str(e)+'.png')
                if is_wandb:
                    wandb.log({'Confusion_Matrix': df_confmat.to_numpy()}, step=e)
    # return required for backwards compatibility with the old API
    # TODO(team-ml) clean up and remove return
    return loss_results


supported_models = ['baseline_cnn1d', 'AMCNet', 'ResNet', 'LSTM', 'MCformer']
supported_outmode = ['real', 'complex', 'real_invdim', 'real_ampphase']

if __name__ == "__main__":
    import argparse

    sed = 0

    # tf.random.set_seed(sed)
    random.seed(sed)
    np.random.seed(sed)
    torch.manual_seed(sed)
    # torch.seed(sed)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--noise", type=bool, default=True, help="Specify if noise needs to be applied or not during training")
    parser.add_argument("--snr_db", nargs='+', default=[30], help="SNR levels to be considered during training. "
                                                                  "It's possible to define multiple noise levels to be "
                                                                  "chosen at random during input slices generation.")
    parser.add_argument("--useRay", action='store_true', default=False, help="Run with Ray's Trainer function")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--address", required=False, type=str, help="the address to use for Ray")
    parser.add_argument("--test", action="store_true", default=False, help="Testing the model")
    parser.add_argument("--cp_path", default='./', help='Path to the checkpoint to save/load the model.')
    parser.add_argument("--protocols", nargs='+', default=['802_11ax', '802_11b_upsampled', '802_11n', '802_11g'],
                        choices=['802_11ax', '802_11b', '802_11b_upsampled', '802_11n', '802_11g'], help="Specify the protocols/classes to be included in the training")
    parser.add_argument("--channel", default=None, choices=['TGn', 'TGax', 'Rayleigh', 'relative', 'random', 'None', None], help="Specify the channel models to apply during data generation. ")
    parser.add_argument('--raw_path', default='../data/DATASET1_1', help='Path where raw signals are stored.')
    parser.add_argument('--slicelen', default=128, type=int, help='Signal slice size')
    parser.add_argument('--overlap_ratio', default=0.5, help='Overlap ratio for slices generation')
    parser.add_argument('--postfix', default='', help='Postfix to append to dataset file.')
    parser.add_argument('--raw_data_ratio', default=1.0, type=float, help='Specify the ratio of examples per class to consider while training/testing')
    parser.add_argument("--normalize", action='store_true', default=False, help="Use a layer norm as a first layer.")
    parser.add_argument('--model', choices=supported_models, default='baseline_cnn1d', help='Model to be used for training')
    parser.add_argument('--out_mode', choices=supported_outmode, default='real', help='Specify data generator output format')
    parser.add_argument('--debug', default=False, action='store_true', help='It will force run on cpu and disable Wandb.')
    
    parser.add_argument("--Epochs", type=int, default=5)
    parser.add_argument("--Learning_rate", type=float, default=0.001)
    parser.add_argument("--Batch_size", type=int, default=512)



    args, _ = parser.parse_known_args()

    print('Apply noise:', args.noise)
    args.channel = args.channel if args.channel != 'None' else None
    protocols = args.protocols
    ds_train = TPrimeDataset(protocols,
                           ds_path=args.raw_path,
                           ds_type='train',
                           snr_dbs=args.snr_db,
                           slice_len=args.slicelen,
                           slice_overlap_ratio=float(args.overlap_ratio),
                           raw_data_ratio=args.raw_data_ratio,
                           file_postfix=args.postfix,
                           override_gen_map=False,
                           apply_wchannel=args.channel,
                           apply_noise=args.noise,
                           out_mode=args.out_mode,
                           seed = sed)
    ds_test = TPrimeDataset(protocols,
                          ds_path=args.raw_path,
                          ds_type='test',
                          snr_dbs=args.snr_db,
                          slice_len=args.slicelen,
                          slice_overlap_ratio=float(args.overlap_ratio),
                          raw_data_ratio=args.raw_data_ratio,
                          file_postfix=args.postfix,
                          override_gen_map=False,    # it will use the same as above call
                          apply_wchannel=args.channel,
                          apply_noise=args.noise,
                          out_mode=args.out_mode,
                          seed = sed)

    if not os.path.isdir(args.cp_path):
        os.makedirs(args.cp_path)

    if args.debug:
        device = "cpu"
        is_wandb = False
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_wandb = True


    ds_info = ds_train.info()
    Nclass = ds_info['nclasses']

    assert(args.model in supported_models)



    train_config = {

        'epochs': args.Epochs,
        'lr': args.Learning_rate,
        'batch_size': args.Batch_size,
        'pytorch_model': args.model,
        'Nclass': Nclass,
        'useRay': args.useRay,
        # TODO: fix this, currently it's not working with Ray because the dataset gets replicated among workers
        'slice_len': ds_info['slice_len'],
        'normalize': args.normalize,
        'num_feats': 1,
        'num_chans': ds_info['nchans'],
        'device': device,
        'cp_path': args.cp_path,
        'class_labels': protocols,
        'wchannel': args.channel if args.channel is not None else 'nochan',
        'postfix': args.postfix,
        'snr_dB': args.snr_db[0],
        'is_wandb': is_wandb
    }

    """
    if not train_config['isDebug']:
        import ray

        ray.init(address=args.address)
        trainer = TorchTrainer(
            train_func,
            train_loop_config=train_config,
            scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu),
        )
        result = trainer.fit()
        print(f"Results: {result.metrics}")
    else:
        train_func(train_config)
    """

    exp_config = {  # Experiment configuration for tracking
        "Dataset": "1_1",
        "Dataset_Ratio": args.raw_data_ratio,
        "Architecture": args.model,
        "Wireless channel": args.channel,
        "Snr (dbs)": args.snr_db,
        "Learning rate": train_config['lr'],
        "Batch size": train_config['batch_size'],
        "Epochs": train_config['epochs'],
        "Slice length": args.slicelen,
    }
    if is_wandb:
        wandb.init(project="RF_Baseline_CNN1D", config=exp_config)
        wandb.run.name = f'{args.snr_db[0]} dBs {args.channel} sl:{ds_info["slice_len"]}'

    epochs_loss = train_func(train_config)
    pickle.dump(epochs_loss, open(os.path.join(args.cp_path, 'epochs_loss.pkl'), 'wb'))
    print(epochs_loss)
    if is_wandb:
        wandb.finish()


