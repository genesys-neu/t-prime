import os
import numpy as np
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir)))
import sys
sys.path.append(proj_root_dir)
from dstl.preprocessing.DSTL_dataset import DSTLDataset
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

def train_epoch(dataloader, model, loss_fn, optimizer, use_ray=False):
    if use_ray:
        size = len(dataloader.dataset) // session.get_world_size()
    else:
        size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader), desc="Training epochs.."):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Cached files: ", len(dataloader.dataset.signal_cache.cache.keys()))

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

    return test_loss, conf_matrix

def train_func(config: Dict):
    global_model = config['pytorch_model']
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
    model = global_model(classes=Nclass, numChannels=num_channels, slice_len=slice_len)
    if use_ray:
        model = train.torch.prepare_model(model)
    else:
        model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
    loss_results = []
    best_loss = np.inf
    for e in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer, use_ray)
        loss, conf_matrix = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass, use_ray=use_ray)
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
                pickle.dump(conf_matrix, open(os.path.join(logdir, 'conf_matrix.best.pkl'), 'wb'))
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(logdir,'model.best.pt'))

    # return required for backwards compatibility with the old API
    # TODO(team-ml) clean up and remove return
    return loss_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", action='store_true', default=False, help="Specify if noise needs to be applied or not during training")
    parser.add_argument("--snr_db", nargs='+', default=[30], help="SNR levels to be considered during training. "
                                                                  "It's possible to define multiple noise levels to be "
                                                                  "chosen at random during input slices generation.")
    parser.add_argument("--useRay", action='store_true', default=False, help="Run with Ray's Trainer function")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--address", required=False, type=str, help="the address to use for Ray")
    parser.add_argument("--test", action="store_true", default=False, help="Testing the model")
    parser.add_argument("--cp_path", default='./', help='Path to the checkpoint to save/load the model.')
    parser.add_argument("--protocols", nargs='+', default=['802_11ax', '802_11b', '802_11n', '802_11g'],
                        choices=['802_11ax', '802_11b', '802_11b_upsampled', '802_11n', '802_11g'], help="Specify the protocols/classes to be included in the training")
    parser.add_argument("--channel", default=None, choices=['TGn', 'TGax', 'Rayleigh', 'relative', None], help="Specify the channel models to apply during data generation. ")
    args, _ = parser.parse_known_args()

    protocols = args.protocols
    ds_train = DSTLDataset(protocols, ds_type='train', snr_dbs=args.snr_db, slice_len=128, slice_overlap_ratio=0.5, override_gen_map=True, apply_wchannel=args.channel, apply_noise=args.noise)
    ds_test = DSTLDataset(protocols, ds_type='test', snr_dbs=args.snr_db, slice_len=128, slice_overlap_ratio=0.5, override_gen_map=True, apply_wchannel=args.channel, apply_noise=args.noise)

    if not os.path.isdir(args.cp_path):
        os.makedirs(args.cp_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = {"lr": 1e-3, "batch_size": 512, "epochs": 5}
    ds_info = ds_train.info()
    Nclass = ds_info['nclasses']
    train_config['pytorch_model'] = Baseline_CNN1D
    train_config['Nclass'] = Nclass
    train_config['useRay'] = args.useRay    # TODO: fix this, currently it's not working with Ray because the dataset gets replicated among workers
    train_config['slice_len'] = ds_info['slice_len']
    train_config['num_feats'] = 1
    train_config['num_chans'] = ds_info['nchans']
    train_config['device'] = device
    train_config['cp_path'] = args.cp_path

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

    train_func(train_config)

