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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ray.train.torch import TorchTrainer, TorchPredictor
from ray.air.config import ScalingConfig
import ray.train as train
import pickle
import json
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from sklearn.metrics import confusion_matrix as conf_mat
import wandb
import matplotlib.pyplot as plt
import scipy.signal as sig
from model_transformer import TransformerModel, TransformerModel_v2


# Function to change the shape of obs
# the input is obs with shape (channel, slice)








#TESTING INFERENCING POST TRAINING

def seed_worker(worker_id):
    import random
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq


def train_epoch(dataloader, model, loss_fn, optimizer, use_ray=False):
    if use_ray:
        print("USING RAY")
        size = len(dataloader.dataset) // session.get_world_size()
    else:
        print("NOTRAY")
        size = len(dataloader.dataset)
    model.train()
    correct = 0
    loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        X_shape = X.shape
        # print(X.shape)

        X_np = X[0][0].cpu().numpy()

        # Reconstruct complex signal from concatenated real and imaginary parts
        half_len = X_np.shape[-1] // 2
        x_complex = X_np[:half_len] + 1j * X_np[half_len:]


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

    # for i in range(len(y)):
    #     print(f"Predicted: {pred.argmax(1)[i].item()}, Actual: {y[i].item()}")
    return test_loss, correct, conf_matrix


def train_func(config: Dict):
    global_model = config['pytorch_model']
    print("global_model", global_model)
    batch_size = config["batch_size"]
    lr = config["lr"]
    print("CONFIG", config)
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
    pos_encoder = config["use_positional_enc"]

    if not use_ray:
        worker_batch_size = batch_size
    else:
        worker_batch_size = batch_size // session.get_world_size()

    # Create data loaders
    # multi-process loading; give every worker its own numpy/random stream so augmentations differ
    loader_kw = dict(num_workers=config['loader_workers'], persistent_workers=config['loader_workers'] > 0,
                     worker_init_fn=seed_worker)
    train_dataloader = DataLoader(ds_train, batch_size=worker_batch_size, shuffle=True, **loader_kw)
    test_dataloader = DataLoader(ds_test, batch_size=worker_batch_size, shuffle=True, **loader_kw)
    # print(f"Training dataset size: {len(ds_train)}")
    # print(f"Validation/Test dataset size: {len(ds_test)}")
    # for batch in test_dataloader:
    #     X_batch, y_batch = batch
    #     print("X_batch shape:", X_batch.shape)
    #     print("y_batch shape:", y_batch.shape)
    #     print("X_batch:", X_batch)
    #     print("y_batch:", y_batch)
    #     break
    #     break  # Only inspect the first batch
    if use_ray:
        train_dataloader = train.torch.prepare_data_loader(train_dataloader)
        test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    # Create model
    #print("PARAMTERS FOR INFERENCE", d_model, seq_len, transformer_layers, pos_encoder)
    model = global_model(classes=Nclass, d_model=d_model, seq_len=seq_len, nlayers=transformer_layers, use_pos=pos_encoder)
    if use_ray:
        model = train.torch.prepare_model(model)
    else:
        print('notray')
        model.to(device)
    
    # print(model)
    for name, param in model.named_parameters():
        print(f'{name:20} {param.numel()} {list(param.shape)}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TOTAL PARAMS {total_params}')
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
    loss_results = []
    best_loss = np.inf
    #wandb.watch(model, log_freq=10)
    best_conf_matrix = 0
    # early stopping: stop once val loss hasn't improved by more than min_delta (relative) for `patience` epochs
    patience, min_delta = config['patience'], config['min_delta']
    es_best, es_wait = np.inf, 0
    history = []
    #print(epochs)
    for e in range(epochs):
        print(f"\n--- Epoch {e+1}/{epochs} ---")
        ""


        tr_loss, tr_acc = train_epoch(train_dataloader, model, loss_fn, optimizer, use_ray)
        #wandb.log({'Tr_loss': tr_loss}, step=e)
        #wandb.log({'Tr_acc': tr_acc}, step=e)
        loss, acc, conf_matrix = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass, use_ray=use_ray)
        #wandb.log({'Val_loss': loss}, step=e)
        #wandb.log({'Val_acc': acc}, step=e)
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
            # print('ifbest loss > loss then do this')
            if best_loss > loss:
                best_loss = loss
                best_conf_matrix = conf_matrix
                if not os.path.exists(logdir):
                    os.makedirs(logdir)
                pickle.dump(conf_matrix, open(os.path.join(logdir, 'conf_matrix.best.pkl'), 'wb'))
                model_name = f'model{config["wchannel"]}_{config["snr"]}_lg.pt' if seq_len == 64 else f'model{config["wchannel"]}_{config["snr"]}_sm.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'epoch': e + 1,
                }, os.path.join(logdir, model_name))
            history.append({'epoch': e + 1, 'train_loss': float(tr_loss), 'train_acc': float(tr_acc),
                            'val_loss': float(loss), 'val_acc': float(acc), 'lr': optimizer.param_groups[0]['lr']})
            with open(os.path.join(logdir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=1)
            if loss < es_best * (1 - min_delta):
                es_best, es_wait = loss, 0
            else:
                es_wait += 1
            if patience > 0 and es_wait >= patience:
                print(f"Early stopping at epoch {e+1}: no val-loss improvement > {min_delta:.1%} in {patience} epochs "
                      f"(best {best_loss:.6f})")
                break
    if epochs == 0:
        print('epochs = zero')
        loss, acc, conf_matrix = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass, use_ray=use_ray)

        print(loss, acc, conf_matrix)
    
    #wandb.log({"Num. params": total_params})
    fig = plt.figure(figsize=(8,8))
    # best_conf_matrix = best_conf_matrix.astype('float') / best_conf_matrix.sum(axis=1)[np.newaxis]
    # plt.imshow(best_conf_matrix, interpolation='none', cmap=plt.cm.Blues)
    # #for i in range(best_conf_matrix.shape[0]):
    # #    for j in range(best_conf_matrix.shape[1]):
    # #        plt.text(x=j, y=i,s=best_conf_matrix[i, j], va='center', ha='center', size='xx-large')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.clim(0, 1)
    # tick_marks = np.arange(Nclass)
    # plt.xticks(tick_marks, config['protocols'])
    # plt.yticks(tick_marks, config['protocols'])
    # plt.tight_layout()
    
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title(f"Confusion matrix: {args.snr_db[0]} dBs, channel: {args.wchannel}, slice: {slice_len}, seq.: {seq_len}")
    # plt.show()
    return loss_results, fig

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--snr_db", nargs='+', default=[30], help="SNR levels to be considered during training. "
                                                                  "It's possible to define multiple noise levels to be "
                                                                  "chosen at random during input slices generation.")
    parser.add_argument("--useRay", action='store_true', default=False, help="Run with Ray's Trainer function")
    parser.add_argument("--num-workers", "-n", type=int, default=2, help="Sets number of workers for training.")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--address", required=False, type=str, help="the address to use for Ray")
    parser.add_argument("--test", action="store_true", default=False, help="Testing the model")
    parser.add_argument("--wchannel", default=None, help="Wireless channel to be applied, it can be"
                                                         "TGn, TGax, Rayleigh, relative or random.")
    parser.add_argument('--raw_path', default='../data/DATASET1_1', help='Path where raw signals are stored.')
    parser.add_argument('--postfix', default='', help='Postfix to append to dataset file.')
    parser.add_argument("--cp_path", default='./model_cp', help='Path to the checkpoint to save/load the model.')
    parser.add_argument("--cls_token", action="store_true", default=False, help="Use the Transformer v2")
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for training and validation.")
    parser.add_argument("--Layers", type=int, default=2)
    parser.add_argument("--Epochs", type=int)
    parser.add_argument("--Learning_rate", type=float)
    parser.add_argument("--Batch_size", type=int, default=122)
    parser.add_argument("--Slice_length", type=int, default=128, help="Slice length in which a sequence is divided.")
    parser.add_argument("--Sequence_length", type=int, default=64, help="Sequence length to input to the transformer.")
    parser.add_argument("--Positional_encoder")
    parser.add_argument('--overlap_ratio', type = float, default=0.5, help='Overlap ratio for slices generation')
    parser.add_argument('--channel_path', default='..folder_of_chans/wired', help='folder of channels to apply')
    parser.add_argument('--use_channel_aug', action='store_true', default=False, help='Apply channel augmentation using .mat files in channel_path')
    parser.add_argument("--use_sota", action='store_true', default=False, help="does it use sota")
    parser.add_argument('--sota_type', default='awgn', help='what sota to use')
    parser.add_argument("--use_cfo", action='store_true', default=False, help="does it use cfo offset")
    parser.add_argument('--max_cfo', default='2000', help='what mac cfo to use in hz')
    parser.add_argument('--loader_workers', type=int, default=0, help='DataLoader worker processes')
    parser.add_argument('--patience', type=int, default=0, help='early-stopping patience in epochs (0 = off)')
    parser.add_argument('--min_delta', type=float, default=0.0, help='relative val-loss improvement that resets patience')
    parser.add_argument('--awgn_snr_min', type=float, default=5.0, help='min SNR (dB) for --sota_type=awgn')
    parser.add_argument('--awgn_snr_max', type=float, default=30.0, help='max SNR (dB) for --sota_type=awgn')
    parser.add_argument("--protocols", nargs='+', default=['16QAM', '64QAM', '8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QPSK'],
                        choices=['16QAM', '64QAM', '8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QPSK', '802.11ax', '802.11b', '802.11g', '802.11n',
                                 '5GNR', 'LoRa', 'Nbiot', 'OFDM', 'ssm', 'Zigbee'], help="Specify the protocols/classes to be included in the training")
    # parser.add_argument("--inference_set", default='../inference_sets/inference_set_grid2', help='Path to the inference set after training.')

    args, _ = parser.parse_known_args()
    args.wchannel = args.wchannel if args.wchannel != 'None' else None
    args.Positional_encoder = args.Positional_encoder in {'True', 'true'}
    postfix = '' if not args.cls_token else '_v2'
    args.cp_path = args.cp_path + postfix
    print("ARGS.CP_PATH with postfix", args.cp_path)
    exp_config = { #Experiment configuration for tracking
        "Dataset": "1_1",
        "raw_path": args.raw_path,
        "Architecture": "Transformer_v1" if not args.cls_token else "Transformer_v2",
        "Layers": args.Layers,
        "Wireless channel": args.wchannel,
        "Snr (dbs)": args.snr_db,
        "Epochs": args.Epochs,
        "Learning rate": args.Learning_rate,
        "Batch size": args.Batch_size,
        "Sequence length": args.Sequence_length,
        "Slice length": args.Slice_length,
        "Input field of view": args.Sequence_length*args.Slice_length,
        "Positional encoder": args.Positional_encoder
    }
    protocols =  args.protocols
    # print('awgn [-30,30]')
    print('channel path, use channel aug')
    print(args.channel_path, args.use_channel_aug)  
    print('use sota, sota type')
    print(args.use_sota, args.sota_type)
    print('use cfo, max cfo')
    print(args.use_cfo, args.max_cfo)
    
    print('protocols', args.protocols)
    ds_train = TPrimeDataset_Transformer(protocols=protocols, ds_type='train', file_postfix=args.postfix, ds_path=exp_config["raw_path"], snr_dbs=args.snr_db, seq_len=exp_config["Sequence length"], slice_len=exp_config["Slice length"], slice_overlap_ratio=args.overlap_ratio, raw_data_ratio=args.dataset_ratio,
            override_gen_map=True, apply_wchannel=args.wchannel, transform=chan2sequence, channel_path=args.channel_path, use_channel_aug=args.use_channel_aug, use_sota = args.use_sota, sota_type = args.sota_type, use_cfo = args.use_cfo,max_cfo = args.max_cfo,
            awgn_snr_range=(args.awgn_snr_min, args.awgn_snr_max))
    print("THIS IS ARGS", args)
    ds_test = TPrimeDataset_Transformer(protocols=protocols, ds_type='test', file_postfix=args.postfix, ds_path=exp_config["raw_path"], snr_dbs=args.snr_db, seq_len=exp_config["Sequence length"], slice_len=exp_config["Slice length"], slice_overlap_ratio=args.overlap_ratio, raw_data_ratio=args.dataset_ratio,
            override_gen_map=False, apply_wchannel=args.wchannel, transform=chan2sequence, channel_path=args.channel_path, use_channel_aug=args.use_channel_aug, use_sota = args.use_sota, sota_type = args.sota_type, use_cfo = args.use_cfo,max_cfo = args.max_cfo,
            awgn_snr_range=(args.awgn_snr_min, args.awgn_snr_max))
    
    print("Experiment Configuration:")
    for key, value in exp_config.items():
        print(f"{key}: {value}")
    # Save ds_train as a pickle file for later loading and plotting
    if not os.path.exists(args.cp_path):
        os.makedirs(args.cp_path)
    with open(os.path.join(args.cp_path, "ds_train.pkl"), "wb") as f:
        pickle.dump(ds_train, f)


    if not os.path.isdir(args.cp_path):
        print("MAKING ", args.cp_path)
        os.makedirs(args.cp_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_info = ds_train.info()
    Nclass = ds_info['nclasses']
    print("Nclasses = ", Nclass)
    
    train_config = {
        "lr": exp_config["Learning rate"], 
        "batch_size": exp_config["Batch size"], 
        "epochs": exp_config["Epochs"],
        "pytorch_model": TransformerModel if not args.cls_token else TransformerModel_v2,
        "transformer_layers": exp_config["Layers"],
        "Nclass": Nclass,
        "useRay": args.useRay, # TODO: fix this, currently it's not working with Ray because the dataset gets replicated among workers 
        "seq_len": ds_info["seq_len"],
        "slice_len": ds_info["slice_len"],
        "num_chans": ds_info['nchans'],
        "device": device,
        "cp_path": args.cp_path,
        "use_positional_enc": exp_config["Positional encoder"],
        "protocols": protocols,
        "wchannel": args.wchannel,
        "snr": args.snr_db[0],
        "loader_workers": args.loader_workers,
        "patience": args.patience,
        "min_delta": args.min_delta
        }

    #wandb.init(project="RF_Transformer", config=exp_config)
    #wandb.run.name = f'{args.snr_db[0]} dBs {args.wchannel} sl:{ds_info["slice_len"]} sq:{ds_info["seq_len"]} {postfix}'
    print("Train config", train_config)
    _, conf_matrix = train_func(train_config)

    #wandb.log({"Confusion Matrix": conf_matrix})
    #wandb.finish()
