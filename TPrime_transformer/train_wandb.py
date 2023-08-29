import os
import numpy as np
import sys
sys.path.insert(0, '../')
from preprocessing.TPrime_dataset import TPrimeDataset_Transformer
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
import wandb
import matplotlib.pyplot as plt

from model_transformer import TransformerModel


# Function to change the shape of obs
# the input is obs with shape (channel, slice)
def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq


def train_epoch(dataloader, model, loss_fn, optimizer, use_ray=False):
    if use_ray:
        size = len(dataloader.dataset) // session.get_world_size()
    else:
        size = len(dataloader.dataset)
    model.train()
    correct = 0
    loss = 0
    for batch, (X, y) in enumerate(dataloader):
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
    global_model = config['pytorch_model']
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    Nclass = config["Nclass"]
    use_ray = config['useRay']
    seq_len = config['seq_len']
    slice_len = config['slice_len']
    d_model = 2 * slice_len
    transformer_layers = config["transformer_layers"] 
    num_channels = config['num_chans']
    device = config['device']
    logdir = config['cp_path']

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
    model = global_model(classes=Nclass, d_model=d_model, seq_len=seq_len, nlayers=transformer_layers)
    if use_ray:
        model = train.torch.prepare_model(model)
    else:
        model.to(device)
    
    print(model)
    for name, param in model.named_parameters():
        print(f'{name:20} {param.numel()} {list(param.shape)}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TOTAL                {total_params}')
    wandb.log({'Num. params': total_params})
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
    loss_results = []
    best_loss = np.inf
    wandb.watch(model, log_freq=10)
    best_conf_matrix = 0
    for e in range(epochs):
        tr_loss, tr_acc = train_epoch(train_dataloader, model, loss_fn, optimizer, use_ray)
        wandb.log({'Tr_loss': tr_loss}, step=e)
        wandb.log({'Tr_acc': tr_acc}, step=e)
        loss, acc, conf_matrix = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass, use_ray=use_ray)
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
            if best_loss > loss:
                best_loss = loss
                best_conf_matrix = conf_matrix
                pickle.dump(conf_matrix, open(os.path.join(logdir, 'conf_matrix.best.pkl'), 'wb'))
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(logdir,'model.best.pt'))
    
    fig = plt.figure(figsize=(8,8))
    best_conf_matrix = best_conf_matrix.astype('float') / best_conf_matrix.sum(axis=1)[np.newaxis]
    plt.imshow(best_conf_matrix, interpolation='none', cmap=plt.cm.Blues)
    #for i in range(best_conf_matrix.shape[0]):
    #    for j in range(best_conf_matrix.shape[1]):
    #        plt.text(x=j, y=i,s=best_conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.clim(0, 1)
    tick_marks = np.arange(Nclass)
    config['protocols'][1] = '802_11b_up'
    plt.xticks(tick_marks, config['protocols'])
    plt.yticks(tick_marks, config['protocols'])
    plt.tight_layout()
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"Confusion matrix: {float(args.snr_db[0])} dBs, channel: {args.wchannel}, slice: {slice_len}, seq.: {seq_len}")
    # return required for backwards compatibility with the old API
    # TODO(team-ml) clean up and remove return
    return loss_results, fig

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_db", nargs='+', default=[30], help="SNR levels to be considered during training. "
                                                                  "It's possible to define multiple noise levels to be "
                                                                  "chosen at random during input slices generation.")
    parser.add_argument("--useRay", action='store_true', default=False, help="Run with Ray's Trainer function")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--address", required=False, type=str, help="the address to use for Ray")
    parser.add_argument("--test", action="store_true", default=False, help="Testing the model")
    parser.add_argument("--wchannel", default=None, help="Wireless channel to be applied, it can be"
                                                         "TGn, TGax, Rayleigh or relative.")
    parser.add_argument("--cp_path", default='./', help='Path to the checkpoint to save/load the model.')
    parser.add_argument("--slice_len", default=128, help="Slice length in which a sequence is divided.")
    parser.add_argument("--seq_len", default=64, help="Sequence length to input to the transformer.")
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for training and validation.")
    args, _ = parser.parse_known_args()

    protocols = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
    ds_train = TPrimeDataset_Transformer(protocols=protocols, ds_type='train', snr_dbs=args.snr_db, seq_len = int(args.seq_len), slice_len=int(args.slice_len), raw_data_ratio=args.dataset_ratio, slice_overlap_ratio=0,
                           override_gen_map=True, apply_wchannel=args.wchannel, transform=chan2sequence)
    ds_test = TPrimeDataset_Transformer(protocols=protocols, ds_type='test', snr_dbs=args.snr_db, seq_len = int(args.seq_len), slice_len=int(args.slice_len), raw_data_ratio=args.dataset_ratio, slice_overlap_ratio=0,
                          override_gen_map=False, apply_wchannel=args.wchannel, transform=chan2sequence)

    if not os.path.isdir(args.cp_path):
        os.makedirs(args.cp_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_info = ds_train.info()
    Nclass = ds_info['nclasses']
    train_config = {
        "lr": 1e-4, 
        "batch_size": 122, 
        "epochs":5,
        "pytorch_model": TransformerModel,
        "transformer_layers": 2,
        "Nclass": Nclass,
        "useRay": args.useRay, # TODO: fix this, currently it's not working with Ray because the dataset gets replicated among workers 
        "seq_len": ds_info['seq_len'],
        "slice_len": ds_info['slice_len'],
        "num_chans": ds_info['nchans'],
        "device": device,
        "cp_path": args.cp_path,
        "protocols": protocols
        }

    exp_config = { #Experiment configuration for tracking
        "Dataset": "1_1",
        "Architecture": "Transformer_v1",
        "Layers": train_config["transformer_layers"],
        "Wireless channel": args.wchannel,
        "Snr (dbs)": args.snr_db,
        "Epochs": train_config["epochs"],
        "Learning rate": train_config["lr"],
        "Batch size": train_config["batch_size"],
        "Sequence lenght": train_config["seq_len"],
        "Slice length": train_config["slice_len"],
        "Input field of view": train_config["seq_len"]*train_config["slice_len"],
        "Positional encoder": "False"
    }
    wandb.init(project="RF_Transformer", config=exp_config)
    wandb.run.name = f'{float(args.snr_db[0])} dBs {args.wchannel} sl:{ds_info["slice_len"]} sq:{ds_info["seq_len"]}'

    _, conf_matrix = train_func(train_config)
    wandb.log({"Confusion Matrix": conf_matrix})
    wandb.finish()
