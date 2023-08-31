import argparse
import sys
import os
sys.path.insert(0, '../')
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
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']) # CHANGE FOR FINE TUNE
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
    train_acc = []
    test_acc = []
    best_acc = 0
    best_cm = 0 # best confusion matrix
    epochs_wo_improvement = 0

    if config['RMSNorm']:
        RMSNorm_l = RMSNorm(model='Transformer')
    else:
        RMSNorm_l = None

    # Training loop
    for epoch in range(config['epochs']):
        acc, loss = train(model, criterion, optimizer, train_dataloader, RMSnorm_layer=RMSNorm_l)
        train_acc.append(acc)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')
        acc, loss, conf_matrix = validate(model, criterion, test_dataloader, config['nClasses'], RMSnorm_layer=RMSNorm_l)
        test_acc.append(acc)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (test)')
        scheduler.step(loss)
        epochs_wo_improvement += 1
        if acc > best_acc:
            best_acc = acc
            epochs_wo_improvement = 0
            # Save model and metrics
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(PATH, MODEL_NAME + '.pt')) # + '_' + OTA_DATASET + '_' + TEST_FLAG + '_' + RMS_FLAG + NOISE_FLAG + '_ft.pt'))
            best_cm = conf_matrix
        if epochs_wo_improvement > 12: # early stopping
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
    prot_display = ['ax', 'b', 'n', 'g'] #PROTOCOLS
    if len(PROTOCOLS) > 4: # We need to add noise class
        prot_display.append('noise')
    #prot_display[1] = '802_11b'
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=prot_display)
    disp.plot(cmap="Blues", values_format='.2f')
    disp.ax_.get_images()[0].set_clim(0, 100)
    plt.title(f'Conf. Matrix (%): Total Acc. {(best_acc):>0.1f}%')
    plt.savefig(f"./training/Results_finetune_{MODEL_NAME}_ft.{OTA_DATASET}.{TEST_FLAG}.{RMS_FLAG}{NOISE_FLAG}.pdf")
    plt.clf()
    print('-----------------------------------------------------------------')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='../TPrime_transformer/model_cp', help='Path to the trained model or to where to save the trained model from scratch with model name included')
    parser.add_argument("--ds_path", default='../data', help='Path to the over the air datasets')
    parser.add_argument("--datasets", nargs='+', required=True, help="Dataset name to be used for training or test")
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for training and validation")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for fine-tuning and inference")
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
    MODEL_NAME = get_model_name(args.model_path)
    PATH = '/'.join(args.model_path.split('/')[0:-1])
    PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
    CHANNELS = ['None', 'TGn', 'TGax', 'Rayleigh']
    TEST_FLAG = 'rsg' if args.test_mode == 'random_sampling' else 'fut'
    RMS_FLAG = 'RMSn' if args.RMSNorm else ''
    NOISE_FLAG = '_bckg' if args.back_class else ''
    if args.back_class:
        PROTOCOLS.append('noise') 
    OTA_DATASET = args.ota_dataset
    train_config = {
        'batchSize': 122,
        'lr': 0.00002,
        'epochs': 30,
        'nClasses': len(PROTOCOLS),
        'RMSNorm': args.RMSNorm
    }
    font = {'size': 15}
    plt.rc('font', **font)

    datasets = args.datasets
    ds_train = []
    ds_test = []
    # Load model
    if args.transformer == 'CNN':
        global_model = Baseline_CNN1D
        model = global_model(classes=len(PROTOCOLS), numChannels=2, slice_len=512)
        for ds in datasets:
            ds_train.append(TPrimeDataset(PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='train', slice_len=512, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                            raw_data_ratio=args.dataset_ratio, file_postfix='', override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, add_noise=args.back_class))
            ds_test.append(TPrimeDataset(PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', slice_len=512, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                raw_data_ratio=args.dataset_ratio, file_postfix='', override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, add_noise=args.back_class))
    else:
        # choose correct version
        if args.transformer_version == 'v1':
            global_model = TransformerModel
        else: # v2
            global_model = TransformerModel_v2
        # choose correct size
        if args.transformer == "sm":
            model = global_model(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
            # Load over the air dataset
            for ds in datasets:
                ds_train.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='train', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                              raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
        else: # lg
            model = global_model(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
            for ds in datasets:
                ds_train.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='train', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                              raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
    # concat all loaded datasets
    ds_train = torch.utils.data.ConcatDataset(ds_train)
    if not args.test:
        ds_test = torch.utils.data.ConcatDataset(ds_test)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if args.retrain: # Load pretrained model
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        except:
            raise Exception("The model you provided does not correspond with the selected architecture. Please revise and try again.")
    if args.use_gpu:
        model.cuda()

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
                f"\n\nTest Error for dataset {args.datasets[ds_ix]}: \n "
                f"Accuracy: {(100 * correct):>0.1f}%, "
                f"Avg loss: {test_loss:>8f} \n"
            )
            conf_matrix = conf_matrix.astype('float')
            for r in range(conf_matrix.shape[0]):  # for each row in the confusion matrix
                sum_row = np.sum(conf_matrix[r, :])
                conf_matrix[r, :] = conf_matrix[r, :] / sum_row  * 100.0 # compute in percentage
            conf_matrix[np.isnan(conf_matrix)] = 0
            # plt.figure(figsize=(10,7))
            prot_display = ['ax', 'b', 'n', 'g']#PROTOCOLS
            if len(PROTOCOLS) > 4: # We need to add noise class
                prot_display.append('noise')
            #prot_display[1] = '802_11b'
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=prot_display)
            disp.plot(cmap="Blues", values_format='.2f')
            disp.ax_.get_images()[0].set_clim(0, 100)
            plt.title(f'Conf. Matrix (%): Total Acc. {(100 * correct):>0.1f}%')
            plt.savefig(f"./Results_finetune_{MODEL_NAME}.{args.datasets[ds_ix]}.{TEST_FLAG}.{RMS_FLAG}{NOISE_FLAG}.pdf")
            plt.clf()
            print(f'Confusion matrix (%) for {args.datasets[ds_ix]}')
            print(np.around(conf_matrix, decimals=2))
            print('-------------------------------------------')
            print('-------------------------------------------')
        
        # Global confusion matrix for all test datasets if more than one provided
        if len(args.datasets) > 1:
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
            plt.savefig(f"./Results_finetune_{MODEL_NAME}.{OTA_DATASET}.{TEST_FLAG}.{RMS_FLAG}{NOISE_FLAG}.pdf")
            plt.clf()
            print(f'Global Confusion Matrix (%) for {OTA_DATASET}')
            print(np.around(global_conf_matrix, decimals=2))
            print('-------------------------------------------')
    else:
        # Fine-tune the provided model with the new data
        finetune(model, train_config)
