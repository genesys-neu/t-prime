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

def train(model, criterion, optimizer, dataloader):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    total_loss = 0
    for batch, (X, y) in tqdm(enumerate(dataloader), desc="Training epochs.."):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction error
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
    print(f"Train Error: \n "
          f"Accuracy: {(100 * correct):>0.1f}%, "
          )
    return correct, total_loss

def validate(model, criterion, dataloader, nclasses):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    conf_matrix = np.zeros((nclasses, nclasses))
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X.float())
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_cpu = y.to('cpu')
            pred_cpu = pred.to('cpu')
            conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(nclasses)))
    test_loss /= size
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )

    return correct, test_loss, conf_matrix

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
    # Training loop
    for epoch in range(config['epochs']):
        acc, loss = train(model, criterion, optimizer, train_dataloader)
        train_acc.append(acc)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')
        acc, loss, conf_matrix = validate(model, criterion, test_dataloader, config['nClasses'])
        test_acc.append(acc)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (test)')
        if acc > best_acc:
            best_acc = acc
            # Save model and metrics
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(PATH, MODEL_NAME + '_ft.pt'))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='./model_cp', help='Path to the trained model.')
    parser.add_argument("--ds_path", default='', help='Path to the over the air dataset.')
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for training and validation.")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for fine-tuning and inference")
    parser.add_argument("--transformer_version", default=None, required=False, choices=["v1", "v2"], help='Architecture of the model that will be \
                        finetuned. Options are v1 and v2. These refer to the two Transformer-based architectures available, without or with [CLS] token.')
    parser.add_argument("--transformer", default="CNN", choices=["sm", "lg"], help="Size of transformer to use, options available are small and \
                        large. If not defined CNN architecture will be used.")
    parser.add_argument("--test", default=False, action='store_true', help="If present, we just test the provided model on OTA data.")
    args, _ = parser.parse_known_args()

    # Config
    MODEL_NAME = get_model_name(args.model_path)
    PATH = '/'.join(args.model_path.split('/')[0:-1])
    PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
    CHANNELS = ['None', 'TGn', 'TGax', 'Rayleigh']
    train_config = {
        'batchSize': 122,
        'lr': 0.00002,
        'epochs': 5,
        'nClasses': 4
    }

    # Load model
    if args.transformer == 'CNN':
        global_model = Baseline_CNN1D
        model = global_model(classes=len(PROTOCOLS), numChannels=2, slice_len=512)
        ds_train = DSTLDataset(PROTOCOLS, ds_path=args.ds_path, ds_type='train', snr_dbs=args.snr_db, slice_len=512, slice_overlap_ratio=0,
                           raw_data_ratio=args.dataset_ratio, file_postfix='', override_gen_map=True, ota=True, apply_wchannel=None, apply_noise=False)
        ds_test = DSTLDataset(PROTOCOLS, ds_path=args.ds_path, ds_type='test', snr_dbs=args.snr_db, slice_len=args.slicelen, slice_overlap_ratio=0, 
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
            ds_train = DSTLDataset_Transformer(protocols=PROTOCOLS, ds_path=args.ds_path, ds_type='train', snr_dbs=args.snr_db, seq_len=24, slice_len=64, slice_overlap_ratio=0, 
                                               raw_data_ratio=args.dataset_ratio, override_gen_map=True, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence)
            ds_test = DSTLDataset_Transformer(protocols=PROTOCOLS, ds_path=args.ds_path, ds_type='test', snr_dbs=args.snr_db, seq_len=24, slice_len=64, slice_overlap_ratio=0, 
                                              raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence)
        else: # lg
            model = global_model(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
            ds_train = DSTLDataset_Transformer(protocols=PROTOCOLS, ds_path=args.ds_path, ds_type='train', snr_dbs=args.snr_db, seq_len=64, slice_len=128, slice_overlap_ratio=0, 
                                               raw_data_ratio=args.dataset_ratio, override_gen_map=True, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence)
            ds_test = DSTLDataset_Transformer(protocols=PROTOCOLS, ds_path=args.ds_path, ds_type='test', snr_dbs=args.snr_db, seq_len=64, slice_len=128, slice_overlap_ratio=0, 
                                              raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        if args.use_gpu:
            model.cuda()
    except:
        raise Exception("The model you provided does not correspond with the selected architecture. Please revise and try again.")
    
    if args.test:
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
            conf_matrix[r, :] = float(conf_matrix[r, :]) / float(sum_row)  * 100.0# compute in percentage

        # plt.figure(figsize=(10,7))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=PROTOCOLS)
        disp.plot()
        plt.savefig(f"Results_finetune_{MODEL_NAME}.{train_config['lr']}.pdf")
        plt.clf()
        print('-------------------------------------------')
        print('-------------------------------------------')
        print('Global confusion matrix (%) (all trials)')
        print(conf_matrix)
        
    else:
        # Fine-tune the provided model with the new data
        finetune(model, train_config)