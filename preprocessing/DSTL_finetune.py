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
from DSTL_dataset import DSTLDataset, DSTLDataset_Transformer
from dstl_transformer.model_transformer import TransformerModel, TransformerModel_v2
from cnn_baseline.model_cnn1d import Baseline_CNN1D
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
    total_loss /= size
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
    test_loss /= size
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
        if acc > best_acc:
            best_acc = acc
            # Save model and metrics
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(PATH, MODEL_NAME + '_' + OTA_DATASET + '_' + TEST_FLAG + '_ft.pt'))
            best_cm = conf_matrix
    best_cm = best_cm.astype('float')
    for r in range(best_cm.shape[0]):  # for each row in the confusion matrix
        sum_row = np.sum(best_cm[r, :])
        best_cm[r, :] = best_cm[r, :] / sum_row  * 100.0 # compute in percentage
    print('------------------- Best confusion matrix (%) -------------------')
    print(best_cm)
    prot_display = PROTOCOLS
    prot_display[1] = '802_11b'
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=prot_display)
    disp.plot()
    disp.ax_.get_images()[0].set_clim(0, 100)
    plt.savefig(f"Results_finetune_{MODEL_NAME}_ft.{OTA_DATASET}.{TEST_FLAG}.{config['lr']}.pdf")
    plt.clf()
    print('-----------------------------------------------------------------')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='./model_cp', help='Path to the trained model or where to save the trained from scratch version \
                        and under which name.')
    parser.add_argument("--ds_path", default='', help='Path to the over the air dataset.')
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for training and validation.")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for fine-tuning and inference")
    parser.add_argument("--transformer_version", default=None, required=False, choices=["v1", "v2"], help='Architecture of the model that will be \
                        finetuned. Options are v1 and v2. These refer to the two Transformer-based architectures available, without or with [CLS] token.')
    parser.add_argument("--transformer", default="CNN", choices=["sm", "lg"], help="Size of transformer to use, options available are small and \
                        large. If not defined CNN architecture will be used.")
    parser.add_argument("--test_mode", default="random_sampling", choices=["random_sampling", "inference"], help="Get test from separate files (inference) or \
                        a random sampling of dataset indexes (random_sampling).")
    parser.add_argument("--retrain", action='store_true', default=False, help="Do not load any model and just train from scratch. Model name will be \
                        taken from the model_path given.")
    parser.add_argument("--ota_dataset", default='', help="Flag to add in results name to identify experiment.")
    parser.add_argument("--test", default=False, action='store_true', help="If present, we just test the provided model on OTA data.")
    parser.add_argument("--RMSNorm", default=False, action='store_true', help="If present, we apply RMS normalization on input signals while training and testing")
    args, _ = parser.parse_known_args()

    # Config
    MODEL_NAME = get_model_name(args.model_path)
    PATH = '/'.join(args.model_path.split('/')[0:-1])
    PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
    CHANNELS = ['None', 'TGn', 'TGax', 'Rayleigh']
    TEST_FLAG = 'rsg' if args.test_mode == 'random_sampling' else 'inf'
    OTA_DATASET = args.ota_dataset
    train_config = {
        'batchSize': 122,
        'lr': 0.00002,
        'epochs': 100,
        'nClasses': 4,
        'RMSNorm': args.RMSNorm
    }

    # Load model
    if args.transformer == 'CNN':
        global_model = Baseline_CNN1D
        model = global_model(classes=len(PROTOCOLS), numChannels=2, slice_len=512)
        ds_train = DSTLDataset(PROTOCOLS, ds_path=args.ds_path, ds_type='train', slice_len=512, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                           raw_data_ratio=args.dataset_ratio, file_postfix='', override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False)
        ds_test = DSTLDataset(PROTOCOLS, ds_path=args.ds_path, ds_type='test', slice_len=512, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                            raw_data_ratio=args.dataset_ratio, file_postfix='', override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False)
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
            ds_train = DSTLDataset_Transformer(protocols=PROTOCOLS, ds_path=args.ds_path, ds_type='train', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence)
            ds_test = DSTLDataset_Transformer(protocols=PROTOCOLS, ds_path=args.ds_path, ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                              raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence)
        else: # lg
            model = global_model(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
            ds_train = DSTLDataset_Transformer(protocols=PROTOCOLS, ds_path=args.ds_path, ds_type='train', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence)
            ds_test = DSTLDataset_Transformer(protocols=PROTOCOLS, ds_path=args.ds_path, ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                              raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if not args.retrain: # Load pretrained model
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        except:
            raise Exception("The model you provided does not correspond with the selected architecture. Please revise and try again.")
    if args.use_gpu:
        model.cuda()

    if args.test and not args.retrain:
        # Use the loaded model to do inference over the OTA dataset
        model.to(device)
        model.eval()
        # validation loop through test data
        test_dataloader = DataLoader(ds_test, batch_size=train_config['batchSize'], shuffle=True)
        size = len(test_dataloader.dataset)
        criterion = nn.NLLLoss()
        test_loss, correct = 0, 0
        conf_matrix = np.zeros((train_config['nClasses'], train_config['nClasses']))
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X.float())
                test_loss += criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                y_cpu = y.to('cpu')
                pred_cpu = pred.to('cpu')
                conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(train_config['nClasses'])))
        test_loss /= size
        correct /= size
        # report accuracy and save confusion matrix
        print(
            f"Test Error: \n "
            f"Accuracy: {(100 * correct):>0.1f}%, "
            f"Avg loss: {test_loss:>8f} \n"
        )
        conf_matrix = conf_matrix.astype('float')
        for r in range(conf_matrix.shape[0]):  # for each row in the confusion matrix
            sum_row = np.sum(conf_matrix[r, :])
            conf_matrix[r, :] = conf_matrix[r, :] / sum_row  * 100.0# compute in percentage

        # plt.figure(figsize=(10,7))
        prot_display = PROTOCOLS
        prot_display[1] = '802_11b'
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=prot_display)
        disp.plot()
        disp.ax_.get_images()[0].set_clim(0, 100)
        plt.savefig(f"Results_finetune_{MODEL_NAME}.{OTA_DATASET}.{TEST_FLAG}.{train_config['lr']}.pdf")
        plt.clf()
        print('-------------------------------------------')
        print('-------------------------------------------')
        print('Global confusion matrix (%)')
        print(conf_matrix)
        
    else:
        # Fine-tune the provided model with the new data
        finetune(model, train_config)
