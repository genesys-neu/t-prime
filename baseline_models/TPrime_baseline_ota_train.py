import os
import numpy as np
import sys
sys.path.insert(0, '../')
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix as conf_mat
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from preprocessing.TPrime_dataset import TPrimeDataset
from baseline_models.model_cnn1d import Baseline_CNN1D
from baseline_models.model_AMCNet import AMC_Net
from baseline_models.model_ResNet import ResNet
from baseline_models.model_LSTM import LSTM_ap
from baseline_models.model_MCFormer import MCformer
from preprocessing.model_rmsnorm import RMSNorm

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def get_model_name(name):
    """Extract model name from file path"""
    name = name.split("/")[-1]
    return '.'.join(name.split(".")[0:-1])

def train_epoch(model, criterion, optimizer, dataloader, device, RMSnorm_layer=None):
    """Train model for one epoch"""
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    total_loss = 0
    
    for batch, (X, y) in tqdm(enumerate(dataloader), desc="Training"):
        X = X.to(device)
        y = y.to(device)
        
        # Apply RMS normalization if specified
        if RMSnorm_layer is not None:
            X = RMSnorm_layer(X)
            
        # Forward pass
        pred = model(X.float())
        loss = criterion(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        
        if batch % 50 == 0:
            loss_val, current = loss.item(), batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
            
    total_loss /= len(dataloader)
    correct /= size
    return correct * 100.0, total_loss

def validate_epoch(model, criterion, dataloader, nclasses, device, RMSnorm_layer=None):
    """Validate model for one epoch"""
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    conf_matrix = np.zeros((nclasses, nclasses))
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            if RMSnorm_layer is not None:
                X = RMSnorm_layer(X)
                
            pred = model(X.float())
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            y_cpu = y.to('cpu')
            pred_cpu = pred.to('cpu')
            conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(nclasses)))
            
    test_loss /= len(dataloader)
    correct /= size
    return correct * 100.0, test_loss, conf_matrix

def create_model(model_name, nclasses, slice_len=512, normalize=False):
    """Create model based on model name"""
    if model_name == 'baseline_cnn1d':
        return Baseline_CNN1D(classes=nclasses, numChannels=2, slice_len=slice_len, normalize=normalize)
    elif model_name == 'AMCNet':
        return AMC_Net(num_classes=nclasses, sig_len=slice_len)
    elif model_name == 'ResNet':
        return ResNet(num_classes=nclasses, num_samples=slice_len, iq_dim=2, kernel_size=3, pool_size=2)
    elif model_name == 'LSTM':
        return LSTM_ap(input_shape=[slice_len, 2], hidden_size=128, output_size=nclasses, num_layers=2)
    elif model_name == 'MCformer':
        return MCformer(num_classes=nclasses)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_loss_function(model_name):
    """Get appropriate loss function for the model"""
    if model_name == 'baseline_cnn1d':
        return nn.NLLLoss()
    else:
        return nn.CrossEntropyLoss()

def get_slice_length(model_name):
    """Get recommended slice length for each model"""
    slice_lengths = {
        'baseline_cnn1d': 8192,
        'AMCNet': 8192,
        'ResNet': 1024,
        'LSTM': 512,
        'MCformer': 2048
    }
    return slice_lengths.get(model_name, 512)

def get_batch_size(model_name):
    """Get recommended batch size for each model"""
    batch_sizes = {
        'baseline_cnn1d': 128,
        'AMCNet': 512,
        'ResNet': 128,
        'LSTM': 128,
        'MCformer': 64
    }
    return batch_sizes.get(model_name, 512)

def train_baseline_model(model, config, ds_train, ds_test, device):
    """Main training function for baseline models"""
    # Create data loaders
    train_dataloader = DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=config['batch_size'], shuffle=True)

    print(f'Initiating training for {config["model_name"]}...')
    print(f'Model parameters: {count_parameters(model):,}')
    
    # Define loss, optimizer and scheduler
    criterion = get_loss_function(config['model_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.000001, verbose=True)
    
    train_acc = []
    test_acc = []
    best_acc = 0
    best_cm = None
    epochs_wo_improvement = 0

    # Initialize RMSNorm if needed
    RMSNorm_layer = None
    if config['RMSNorm']:
        if config['model_name'] in ['baseline_cnn1d', 'AMCNet', 'ResNet']:
            RMSNorm_layer = RMSNorm(model='CNN')
        else:
            RMSNorm_layer = RMSNorm(model='Transformer')  # fallback

    # Training loop
    for epoch in range(config['epochs']):
        acc, loss = train_epoch(model, criterion, optimizer, train_dataloader, device, RMSNorm_layer)
        train_acc.append(acc)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')
        
        acc, loss, conf_matrix = validate_epoch(model, criterion, test_dataloader, config['nClasses'], device, RMSNorm_layer)
        test_acc.append(acc)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (test)')
        
        scheduler.step(loss)
        epochs_wo_improvement += 1
        
        if acc > best_acc:
            best_acc = acc
            epochs_wo_improvement = 0
            # Save best model
            model_save_name = f"{config['model_name']}_{config['ota_dataset']}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy': acc,
                'config': config
            }, os.path.join(config['model_path'], model_save_name))
            best_cm = conf_matrix
            
        if epochs_wo_improvement > 15:  # Early stopping
            print('------------------------------------')
            print(f'Early termination at epoch: {epoch+1}')
            print('------------------------------------')
            break

    # Plot and save results
    if best_cm is not None:
        save_results(best_cm, best_acc, config)
    
    return best_acc

def save_results(conf_matrix, accuracy, config):
    """Save confusion matrix and results"""
    # Normalize confusion matrix
    conf_matrix = conf_matrix.astype('float')
    for r in range(conf_matrix.shape[0]):
        sum_row = np.sum(conf_matrix[r, :])
        if sum_row > 0:
            conf_matrix[r, :] = conf_matrix[r, :] / sum_row * 100.0
    
    conf_matrix[np.isnan(conf_matrix)] = 0
    
    print('------------------- Best confusion matrix (%) -------------------')
    print(np.around(conf_matrix, decimals=2))
    
    # Create protocol display labels
    prot_display = ['ax', 'b', 'n', 'g']
    if len(config['protocols']) > 4:  # Add noise class if present
        prot_display.append('noise')
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=prot_display)
    disp.plot(cmap="Blues", values_format='.2f')
    disp.ax_.get_images()[0].set_clim(0, 100)
    plt.title(f'Conf. Matrix (%): {config["model_name"]} - Acc. {accuracy:.1f}%')
    
    # Save plot
    results_dir = './results_ota'
    os.makedirs(results_dir, exist_ok=True)
    
    flags = f"{config['test_flag']}.{config['rms_flag']}{config['noise_flag']}"
    filename = f"Results_{config['model_name']}.{config['ota_dataset']}.{flags}.pdf"
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()
    print(f'Results saved to: {os.path.join(results_dir, filename)}')

def test_baseline_model(model, config, ds_test, device):
    """Test a trained baseline model"""
    print(f'Testing {config["model_name"]} on OTA data...')
    
    # Initialize RMSNorm if needed
    RMSNorm_layer = None
    if config['RMSNorm']:
        if config['model_name'] in ['baseline_cnn1d', 'AMCNet', 'ResNet']:
            RMSNorm_layer = RMSNorm(model='CNN')
        else:
            RMSNorm_layer = RMSNorm(model='Transformer')

    model.eval()
    criterion = get_loss_function(config['model_name'])
    
    # Test on each dataset individually if multiple datasets provided
    if isinstance(ds_test, list):
        global_conf_matrix = np.zeros((config['nClasses'], config['nClasses']))
        global_correct = 0
        global_size = 0
        
        for ds_ix, ds in enumerate(ds_test):
            test_dataloader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False)
            size = len(test_dataloader.dataset)
            global_size += size
            
            test_loss, correct = 0, 0
            conf_matrix = np.zeros((config['nClasses'], config['nClasses']))
            
            with torch.no_grad():
                for X, y in test_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    
                    if RMSNorm_layer is not None:
                        X = RMSNorm_layer(X)
                        
                    pred = model(X.float())
                    test_loss += criterion(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    
                    y_cpu = y.to('cpu')
                    pred_cpu = pred.to('cpu')
                    batch_cm = conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(config['nClasses'])))
                    conf_matrix += batch_cm
                    global_conf_matrix += batch_cm
            
            test_loss /= len(test_dataloader)
            global_correct += correct
            correct /= size
            
            print(f"\nTest results for dataset {config['datasets'][ds_ix]}:")
            print(f"Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
            
            # Save individual dataset results
            config_copy = config.copy()
            config_copy['ota_dataset'] = config['datasets'][ds_ix]
            save_results(conf_matrix, correct * 100, config_copy)
        
        # Global results for all datasets
        if len(config['datasets']) > 1:
            global_correct /= global_size
            print(f"\nGlobal test results for {config['ota_dataset']}:")
            print(f"Accuracy: {(100 * global_correct):>0.1f}%")
            save_results(global_conf_matrix, global_correct * 100, config)
    else:
        # Single dataset testing
        test_dataloader = DataLoader(ds_test, batch_size=config['batch_size'], shuffle=False)
        acc, loss, conf_matrix = validate_epoch(model, criterion, test_dataloader, config['nClasses'], device, RMSNorm_layer)
        print(f"Test Accuracy: {acc:.1f}%, Test Loss: {loss:.4f}")
        save_results(conf_matrix, acc, config)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=['baseline_cnn1d', 'AMCNet', 'ResNet', 'LSTM', 'MCformer'], 
                       required=True, help='Model to be used for training/testing')
    parser.add_argument("--model_path", default='./model_cp_ota', 
                       help='Path to save/load the trained model')
    parser.add_argument("--ds_path", default='../data', 
                       help='Path to the over the air datasets')
    parser.add_argument("--datasets", nargs='+', required=True, 
                       help="Dataset names to be used for training or test")
    parser.add_argument("--dataset_ratio", default=1.0, type=float, 
                       help="Portion of the dataset used for training and validation")
    parser.add_argument("--use_gpu", action='store_true', default=False, 
                       help="Use GPU for training and inference")
    parser.add_argument("--test_mode", default="random_sampling", 
                       choices=["random_sampling", "future"], 
                       help="Test data selection method")
    parser.add_argument("--retrain", action='store_true', default=False, 
                       help="Load and fine-tune existing model")
    parser.add_argument("--ota_dataset", default='', 
                       help="Flag to add in results name to identify experiment")
    parser.add_argument("--test", default=False, action='store_true', 
                       help="Test mode only")
    parser.add_argument("--RMSNorm", default=False, action='store_true', 
                       help="Apply RMS normalization on input signals")
    parser.add_argument("--back_class", default=False, action='store_true', 
                       help="Include background/noise class")
    parser.add_argument("--normalize", default=False, action='store_true', 
                       help="Use layer normalization (for CNN models)")
    parser.add_argument("--batch_size", default=None, type=int, 
                       help="Batch size for training/testing")
    parser.add_argument("--epochs", default=30, type=int, 
                       help="Number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, 
                       help="Learning rate")
    parser.add_argument("--slice_len", default=None, type=int, 
                       help="Override default slice length for the model")
    
    args = parser.parse_args()
    
    # Configuration setup
    PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
    if args.back_class:
        PROTOCOLS.append('noise')
    
    # Set slice length based on model if not specified
    if args.slice_len is None:
        args.slice_len = get_slice_length(args.model)

     # Set batch size based on model if not specified
    if args.batch_size is None:
        args.batch_size = get_batch_size(args.model)

    if args.model == "ResNet":
        args.lr = 0.001
    
    # Create flags for naming
    TEST_FLAG = 'rsg' if args.test_mode == 'random_sampling' else 'fut'
    RMS_FLAG = 'RMSn' if args.RMSNorm else 'noRMS'
    NOISE_FLAG = '_bckg' if args.back_class else ''
    
    config = {
        'model_name': args.model,
        'model_path': args.model_path,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'nClasses': len(PROTOCOLS),
        'protocols': PROTOCOLS,
        'datasets': args.datasets,
        'ota_dataset': args.ota_dataset,
        'test_flag': TEST_FLAG,
        'rms_flag': RMS_FLAG,
        'noise_flag': NOISE_FLAG,
        'RMSNorm': args.RMSNorm,
        'normalize': args.normalize,
        'slice_len': args.slice_len
    }
    
    # Create model directory
    os.makedirs(args.model_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"Loading datasets with slice length: {args.slice_len}")
    ds_train = []
    ds_test = []
    
    font = {'size': 15}
    plt.rc('font', **font)

    for ds in args.datasets:
        # Training dataset
        ds_train.append(TPrimeDataset(
            PROTOCOLS, 
            ds_path=os.path.join(args.ds_path, ds), 
            ds_type='train', 
            slice_len=args.slice_len, 
            slice_overlap_ratio=0.5, 
            test_ratio=0.2, 
            testing_mode=args.test_mode,
            raw_data_ratio=args.dataset_ratio, 
            file_postfix='', 
            override_gen_map=False, 
            ota=True, 
            apply_wchannel=None, 
            apply_noise=False, 
            #add_noise=args.back_class
        ))
        
        # Test dataset
        ds_test.append(TPrimeDataset(
            PROTOCOLS, 
            ds_path=os.path.join(args.ds_path, ds), 
            ds_type='test', 
            slice_len=args.slice_len, 
            slice_overlap_ratio=0.5, 
            test_ratio=0.2, 
            testing_mode=args.test_mode,
            raw_data_ratio=args.dataset_ratio, 
            file_postfix='', 
            override_gen_map=False, 
            ota=True, 
            apply_wchannel=None, 
            apply_noise=False, 
            #add_noise=args.back_class
        ))
    
    # Concatenate datasets
    ds_train = torch.utils.data.ConcatDataset(ds_train)
    if not args.test:
        ds_test = torch.utils.data.ConcatDataset(ds_test)
    
    print(f"Training samples: {len(ds_train)}")
    print(f"Test samples: {len(ds_test) if not isinstance(ds_test, list) else sum(len(ds) for ds in ds_test)}")
    
    # Create model
    model = create_model(args.model, len(PROTOCOLS), args.slice_len, args.normalize)
    model.to(device)
    
    # Load pretrained model if retraining
    if args.retrain:
        model_file = f"{args.model}_{args.ota_dataset}.pt"
        model_path = os.path.join(args.model_path, model_file)
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained model from: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Training from scratch...")
    
    if args.test:
        # Test mode
        test_baseline_model(model, config, ds_test, device)
    else:
        # Training mode
        best_acc = train_baseline_model(model, config, ds_train, ds_test, device)
        print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()