import argparse
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_RESULTS_DIR = os.path.join(_THIS_DIR, "training")
os.makedirs(_RESULTS_DIR, exist_ok=True)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix as conf_mat
import matplotlib.pyplot as plt
from tqdm import tqdm
from TPrime_dataset import TPrimeDataset, TPrimeDataset_Transformer
from TPrime_transformer.model_transformer import TransformerModel, TransformerModel_v2
from baseline_models.model_cnn1d import Baseline_CNN1D
from preprocessing.model_rmsnorm import RMSNorm

# Function to change the shape of obs
# the input is obs with shape (channel, slice)
def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def get_model_name(name):
    name = name.split("/")[-1]
    return '.'.join(name.split(".")[0:-1])


def _pretty_label(name: str) -> str:
    if name.startswith('802_11'):
        label = name.replace('802_11', '').replace('_upsampled', '').replace('_', '')
        return label
    return name


def _autodetect_protocols(root_dir):
    """Return sorted list of class folders contained in root_dir."""
    if root_dir is None or not os.path.isdir(root_dir):
        raise ValueError(f"Dataset path '{root_dir}' is not a valid directory")
    return sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
    ])

# === NEW: checkpoint helpers ===
def _save_checkpoint(path_dir, model_name_prefix, tag, model, optimizer, epoch, acc, loss, extra=None):
    """Write a checkpoint with tag suffix, e.g. <prefix>_last.pt or <prefix>_best.pt."""
    os.makedirs(path_dir, exist_ok=True)
    ckpt_path = os.path.join(path_dir, f"{model_name_prefix}_{tag}.pt")
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "acc": acc,
        "loss": loss,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, ckpt_path)
    print(f"[ckpt] saved {tag}: {ckpt_path}")

def train(model, criterion, optimizer, dataloader, RMSnorm_layer=None):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    total_loss = 0
    for batch, (X, y) in tqdm(enumerate(dataloader), desc="Training epochs.."):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction error
        if not(RMSnorm_layer is None):
            X = RMSnorm_layer(X)
        pred = model(X.float())
        loss = criterion(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        if batch % 50 == 0:
            loss_val, current = loss.item(), batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
    total_loss /= len(dataloader)
    correct /= size
    return correct*100.0, total_loss

def validate(model, criterion, dataloader, nclasses, RMSnorm_layer=None):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    conf_matrix = np.zeros((nclasses, nclasses))
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            if not (RMSnorm_layer is None):
                X = RMSnorm_layer(X)
            pred = model(X.float())
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_cpu = y.to('cpu')
            pred_cpu = pred.to('cpu')
            conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(nclasses)))
    test_loss /= len(dataloader)
    correct /= size
    return correct*100.0, test_loss, conf_matrix

def finetune(model, config):
    # Create data loaders
    train_dataloader = DataLoader(ds_train, batch_size=config['batchSize'], shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=config['batchSize'], shuffle=True)

    print('Initiating fine-tuning...')
    # Define loss, optimizer and scheduler for training
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']) # CHANGE FOR FINE TUNE
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
    train_acc = []
    test_acc = []
    best_acc = 0.0
    best_cm = 0 # best confusion matrix
    epochs_wo_improvement = 0

    if config['RMSNorm']:
        RMSNorm_l = RMSNorm(model='Transformer')
    else:
        RMSNorm_l = None

    # Training loop
    for epoch in range(config['epochs']):
        acc_tr, loss_tr = train(model, criterion, optimizer, train_dataloader, RMSnorm_layer=RMSNorm_l)
        train_acc.append(acc_tr)
        print(f'| epoch {epoch:03d} | train accuracy={acc_tr:.1f}%, train loss={loss_tr:.2f}')

        acc_va, loss_va, conf_matrix = validate(model, criterion, test_dataloader, config['nClasses'], RMSnorm_layer=RMSNorm_l)
        test_acc.append(acc_va)
        print(f'| epoch {epoch:03d} | valid accuracy={acc_va:.1f}%, valid loss={loss_va:.2f} (test)')
        scheduler.step(loss_va)

        # === NEW: always save _last each epoch
        _save_checkpoint(PATH, MODEL_NAME, "last", model, optimizer, epoch, acc_va, loss_va)

        epochs_wo_improvement += 1
        if acc_va > best_acc:
            best_acc = acc_va
            epochs_wo_improvement = 0
            best_cm = conf_matrix

            # === NEW: save _best
            _save_checkpoint(PATH, MODEL_NAME, "best", model, optimizer, epoch, acc_va, loss_va)

            # Back-compat: also save to original single file when new best occurs
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_va,
            }, os.path.join(PATH, MODEL_NAME + '.pt'))
            print(f"[ckpt] saved best (legacy path): {os.path.join(PATH, MODEL_NAME + '.pt')}")

        if epochs_wo_improvement > 12: # early stopping
            print('------------------------------------')
            print('Early termination implemented at epoch:', epoch+1)
            print('------------------------------------')
            break

    best_cm = best_cm.astype('float')
    for r in range(best_cm.shape[0]):  # for each row in the confusion matrix
        sum_row = np.sum(best_cm[r, :])
        best_cm[r, :] = best_cm[r, :] / sum_row  * 100.0 # compute in percentage
    print('------------------- Best confusion matrix (%) -------------------')
    print(np.around(best_cm, decimals=2))
    prot_display = [_pretty_label(p) for p in PROTOCOLS]
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=prot_display)
    disp.plot(cmap="Blues", values_format='.2f')
    disp.ax_.get_images()[0].set_clim(0, 100)
    plt.title(f'Conf. Matrix (%): Total Acc. {(best_acc):>0.1f}%')
    plot_path = os.path.join(_RESULTS_DIR, f"Results_finetune_{MODEL_NAME}_ft.{OTA_DATASET}.{TEST_FLAG}.{RMS_FLAG}{NOISE_FLAG}.pdf")
    plt.savefig(plot_path)
    plt.clf()
    print('-----------------------------------------------------------------')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='../TPrime_transformer/model_cp', help='Path to the trained model or to where to save the trained model from scratch with model name included')
    parser.add_argument("--ds_path", default='../data', help='Path to the over the air datasets')
    parser.add_argument("--raw_path", default=None, help='Path to baseband datasets generated offline (non-OTA)')
    parser.add_argument("--datasets", nargs='+', required=False, help="Dataset name to be used for training or test (when using OTA data)")
    parser.add_argument('--protocols', nargs='+', default=None, help='Protocol/class folder names to use with --raw_path. Autodetected if omitted.')
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for training and validation")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for fine-tuning and inference")
    parser.add_argument("--gpu_device", type=int, default=0, help="GPU index to use when --use_gpu is provided")
    parser.add_argument("--transformer_version", default=None, required=False, choices=["v1", "v2"], help='Architecture of the model that will be \
                        finetuned. Options are v1 and v2. These refer to the two Transformer-based architectures available, without or with [CLS] token')
    parser.add_argument("--transformer", default="CNN", choices=["sm", "lg"], help="Size of transformer to use, options available are small and \
                        large. If not defined CNN architecture will be used")
    parser.add_argument("--test_mode", default="random_sampling", choices=["random_sampling", "future"], help="Get test from separate files (future) or \
                        a random sampling of dataset indexes (random_sampling)")
    parser.add_argument("--retrain", action='store_true', default=False, help="Load the selected model and fine-tune. If this is false the model will be trained from scratch and the model name will be \
                        taken from the model_path given")
    parser.add_argument("--ota_dataset", default='', help="Flag to add in results name to identify experiment.")
    parser.add_argument("--test", default=False, action='store_true', help="If present, just test the provided model on OTA data.")
    parser.add_argument("--RMSNorm", default=False, action='store_true', help="If present, apply RMS normalization on input signals while training and testing")
    parser.add_argument("--back_class", default=False, action='store_true', help="Train/Use model with background or noise class.")
    args, _ = parser.parse_known_args()

    # Config
    INPUT_MODEL_NAME = get_model_name(args.model_path)
    MODEL_NAME = INPUT_MODEL_NAME + ('_finetuned' if args.retrain else '')
    PATH = '/'.join(args.model_path.split('/')[0:-1])

    using_raw_dataset = args.raw_path is not None
    if using_raw_dataset:
        raw_root = os.path.abspath(args.raw_path)
        PROTOCOLS = args.protocols if args.protocols else _autodetect_protocols(raw_root)
        DATASET_LABELS = [os.path.basename(os.path.normpath(raw_root))]
    else:
        if not args.datasets:
            raise ValueError("--datasets must be provided when --raw_path is not used")
        PROTOCOLS = args.datasets
        DATASET_LABELS = PROTOCOLS
    CHANNELS = ['None', 'TGn', 'TGax', 'Rayleigh']
    TEST_FLAG = 'rsg' if args.test_mode == 'random_sampling' else 'fut'
    RMS_FLAG = 'RMSn' if args.RMSNorm else ''
    NOISE_FLAG = '_bckg' if args.back_class else ''
    if args.back_class:
        PROTOCOLS.append('noise')
    OTA_DATASET = args.ota_dataset if args.ota_dataset else (DATASET_LABELS[0] if DATASET_LABELS else '')
    train_config = {
        'batchSize': 122,
        'lr': 0.00002,
        'epochs': 30,
        'nClasses': len(PROTOCOLS),
        'RMSNorm': args.RMSNorm
    }
    font = {'size': 15}
    plt.rc('font', **font)

    datasets = DATASET_LABELS
    ds_train = []
    ds_test = []
    # Load model
    if args.transformer == 'CNN':
        global_model = Baseline_CNN1D
        model = global_model(classes=len(PROTOCOLS), numChannels=2, slice_len=512)
        ds_train.append(TPrimeDataset(PROTOCOLS, ds_path=args.ds_path, ds_type='train', slice_len=512, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                            raw_data_ratio=args.dataset_ratio, file_postfix='', override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, add_noise=args.back_class))
        ds_test.append(TPrimeDataset(PROTOCOLS, ds_path=args.ds_path, ds_type='test', slice_len=512, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                raw_data_ratio=args.dataset_ratio, file_postfix='', override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, add_noise=args.back_class))
    else:
        # choose correct version
        if args.transformer_version == 'v1':
            global_model = TransformerModel
        else: # v2
            global_model = TransformerModel_v2
        # choose correct size
        if args.transformer == "sm":
            seq_len = 24
            slice_len = 64
            model = global_model(classes=len(PROTOCOLS), d_model=64*2, seq_len=seq_len, nlayers=2, use_pos=False)
        else: # lg
            seq_len = 64
            slice_len = 128
            model = global_model(classes=len(PROTOCOLS), d_model=128*2, seq_len=seq_len, nlayers=2, use_pos=False)

        dataset_kwargs_base = dict(
            protocols=PROTOCOLS,
            seq_len=seq_len,
            slice_len=slice_len,
            slice_overlap_ratio=0,
            test_ratio=0.2,
            testing_mode=args.test_mode,
            raw_data_ratio=args.dataset_ratio,
            override_gen_map=False,
            ota=not using_raw_dataset,
            apply_wchannel=None,
            apply_noise=False,
            transform=chan2sequence
        )

        if using_raw_dataset:
            dataset_kwargs_base['ds_path'] = raw_root
        else:
            dataset_kwargs_base['ds_path'] = args.ds_path

        for _ in datasets:
            train_kwargs = dict(dataset_kwargs_base)
            train_kwargs['ds_type'] = 'train'
            ds_train.append(TPrimeDataset_Transformer(**train_kwargs))

            test_kwargs = dict(dataset_kwargs_base)
            test_kwargs['ds_type'] = 'test'
            ds_test.append(TPrimeDataset_Transformer(**test_kwargs))
            if using_raw_dataset:
                break  # raw datasets are instantiated once regardless of number of labels provided
    # concat all loaded datasets
    ds_train = torch.utils.data.ConcatDataset(ds_train)
    if not args.test:
        ds_test = torch.utils.data.ConcatDataset(ds_test)

    if args.use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--use_gpu was set but CUDA is not available on this system")
        gpu_idx = args.gpu_device
        total_gpus = torch.cuda.device_count()
        if gpu_idx < 0 or gpu_idx >= total_gpus:
            raise ValueError(f"Requested GPU index {gpu_idx} is invalid. Available GPUs: 0..{total_gpus-1}")
        device = torch.device(f"cuda:{gpu_idx}")
        print(f"[info] Using GPU device {device}")
    else:
        device = torch.device("cpu")
    if args.retrain: # Load pretrained model
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        except:
            raise Exception("The model you provided does not correspond with the selected architecture. Please revise and try again.")
    model.to(device)

    if args.test and not args.retrain:
        # Use the loaded model to do inference over the OTA dataset
        global_conf_matrix = np.zeros((train_config['nClasses'], train_config['nClasses']))
        global_correct = 0
        global_size = 0
        for ds_ix, ds in enumerate(ds_test):
            # Calculate performance and save matrix
            if train_config['RMSNorm']:
                RMSNorm_l = RMSNorm(model='Transformer')
            else:
                RMSNorm_l = None
            model.to(device)
            model.eval()
            # validation loop through test data
            test_dataloader = DataLoader(ds, batch_size=train_config['batchSize'], shuffle=True)
            size = len(test_dataloader.dataset)
            global_size += size
            criterion = nn.NLLLoss()
            test_loss, correct = 0, 0
            conf_matrix = np.zeros((train_config['nClasses'], train_config['nClasses']))
            with torch.no_grad():
                for X, y in test_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    if not (RMSNorm_l is None):
                        X = RMSNorm_l(X)
                    pred = model(X.float())
                    test_loss += criterion(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    y_cpu = y.to('cpu')
                    pred_cpu = pred.to('cpu')
                    conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(train_config['nClasses'])))
                    global_conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(train_config['nClasses'])))
            test_loss /= len(test_dataloader)
            global_correct += correct
            correct /= size
            # report accuracy and save confusion matrix
            print(
                f"\n\nTest Error for dataset {DATASET_LABELS[ds_ix]}: \n "
                f"Accuracy: {(100 * correct):>0.1f}%, "
                f"Avg loss: {test_loss:>8f} \n"
            )
            conf_matrix = conf_matrix.astype('float')
            for r in range(conf_matrix.shape[0]):  # for each row in the confusion matrix
                sum_row = np.sum(conf_matrix[r, :])
                conf_matrix[r, :] = conf_matrix[r, :] / sum_row  * 100.0 # compute in percentage
            conf_matrix[np.isnan(conf_matrix)] = 0
            prot_display = [_pretty_label(p) for p in PROTOCOLS]
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=prot_display)
            disp.plot(cmap="Blues", values_format='.2f')
            disp.ax_.get_images()[0].set_clim(0, 100)
            plt.title(f'Conf. Matrix (%): Total Acc. {(100 * correct):>0.1f}%')
            plot_path = os.path.join(_RESULTS_DIR, f"Results_finetune_{MODEL_NAME}.{DATASET_LABELS[ds_ix]}.{TEST_FLAG}.{RMS_FLAG}{NOISE_FLAG}.pdf")
            plt.savefig(plot_path)
            plt.clf()
            print(f'Confusion matrix (%) for {DATASET_LABELS[ds_ix]}')
            print(np.around(conf_matrix, decimals=2))
            print('-------------------------------------------')
            print('-------------------------------------------')
        
        # Global confusion matrix for all test datasets if more than one provided
        if len(DATASET_LABELS) > 1:
            global_conf_matrix = global_conf_matrix.astype('float')
            global_correct /= global_size
            print(
                f"\n\nTest Error for dataset {OTA_DATASET}: \n "
                f"Accuracy: {(100 * global_correct):>0.1f}%\n "
            )
            for r in range(global_conf_matrix.shape[0]):  # for each row in the confusion matrix
                sum_row = np.sum(global_conf_matrix[r, :])
                global_conf_matrix[r, :] = global_conf_matrix[r, :] / sum_row  * 100.0 # compute in percentage
            global_conf_matrix[np.isnan(global_conf_matrix)] = 0
            disp = ConfusionMatrixDisplay(confusion_matrix=global_conf_matrix, display_labels=prot_display)
            disp.plot(cmap="Blues", values_format='.2f')
            disp.ax_.get_images()[0].set_clim(0, 100)
            plt.title(f'Global Conf. Matrix (%): Total Acc. {(100 * global_correct):>0.1f}%')
            plot_path = os.path.join(_RESULTS_DIR, f"Results_finetune_{MODEL_NAME}.{OTA_DATASET}.{TEST_FLAG}.{RMS_FLAG}{NOISE_FLAG}.pdf")
            plt.savefig(plot_path)
            plt.clf()
            print(f'Global Confusion Matrix (%) for {OTA_DATASET}')
            print(np.around(global_conf_matrix, decimals=2))
            print('-------------------------------------------')
    else:
        # Fine-tune the provided model with the new data
        finetune(model, train_config)
