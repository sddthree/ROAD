import numpy as np
import torch
import pickle 
from utils.utils_DFS import *
import os
from datasets.dataset_mtl_concat_DFS import save_splits
from sklearn.metrics import roc_auc_score
from models.model_DFS import ROAD_fc_mtl_concat, Total_Loss, NegativeLogLikelihood
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
from lifelines.utils import concordance_index
import time as time2

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    loss_fn = []
    criterion = NegativeLogLikelihood(0).to(device)
    loss_fn.append(criterion)
    cls_loss_fn = nn.CrossEntropyLoss()
    loss_fn.append(cls_loss_fn)
    site_loss_fn = nn.SmoothL1Loss()
    loss_fn.append(site_loss_fn)
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'gene':args.gene, 'cli': args.cli}
    model = ROAD_fc_mtl_concat(**model_dict)
    
#     if len(args.devices > 1):
#         device_ids = args.devices
#         model = torch.nn.DataParallel(model, device_ids=device_ids)

    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    result_dict = {}
    results_dict, cls_train_error, cls_train_auc, _= summary(model, train_loader, args.n_classes)
    result_dict.update(results_dict)
    print('Cls TRAIN error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_train_error, cls_train_auc))
    
    
    results_dict, cls_val_error, cls_val_auc, _= summary(model, val_loader, args.n_classes)
    result_dict.update(results_dict)
    print('Cls Val error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_val_error, cls_val_auc))

    results_dict, cls_test_error, cls_test_auc, acc_loggers= summary(model, test_loader, args.n_classes)
    result_dict.update(results_dict)
    print('Cls Test error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_test_error, cls_test_auc))

#     for i in range(args.n_classes):
#         acc, correct, count = acc_loggers.get_summary(i)
#         print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

#         if writer:
#             writer.add_scalar('final/test_class_{}_tpr'.format(i), acc, 0)


    if writer:
        writer.add_scalar('final/cls_val_error', cls_val_error, 0)
        writer.add_scalar('final/cls_val_auc', cls_val_auc, 0)
        writer.add_scalar('final/cls_test_error', cls_test_error, 0)
        writer.add_scalar('final/cls_test_auc', cls_test_auc, 0)
        writer.add_scalar('final/cls_train_error', cls_train_error, 0)
        writer.add_scalar('final/cls_train_auc', cls_train_auc, 0)

    
    writer.close()
    return result_dict, cls_test_auc, cls_val_auc, cls_train_auc, 1-cls_test_error, 1-cls_val_error, 1-cls_train_error


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    all_time = 0
    cls_train_error = 0.
    cls_train_loss = 0.
    criterion = loss_fn[0]
    cls_loss_fn = loss_fn[1]
    site_loss_fn = loss_fn[2]
    cls_loss_fn_n = torch.tensor([]).to(device)
    site_loss_fn_n = torch.tensor([]).to(device)
    print('\n')
    X = torch.tensor([]).to(device)
    y = torch.tensor([]).to(device)
    e = torch.tensor([]).to(device)
    X_all = torch.tensor([]).to(device)
    y_all = torch.tensor([]).to(device)
    e_all = torch.tensor([]).to(device)
    # 记录开始时间
    start_time = time2.time()
    for batch_idx, (data, label, time, gene, cli, _) in enumerate(loader):
        data =  data.to(device)
        label = label.to(device)
        time = time.float().to(device)
        cli = cli.to(device)
        gene = gene.to(device)
        results_dict = model(data, gene, cli)
        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
#         cls_logger.log(Y_hat, label)
        risk_score = (logits[:,[0,1]].max() * logits[:, 2]).view(1,1)
#         label_loss = cls_loss_fn(logits[:,[0,1]], label)
#         time_loss = site_loss_fn(logits[:, 2], time)
#         cls_loss_fn_n = torch.cat([cls_loss_fn_n, label_loss.view(1,1)])
#         site_loss_fn_n = torch.cat([site_loss_fn_n, time_loss.view(1,1)])
        X = torch.cat([X, risk_score])
        y = torch.cat([y, time.reshape(-1,1)])
        e = torch.cat([e, label.reshape(-1,1)])
        X_all = torch.cat([X_all, risk_score])
        y_all = torch.cat([y_all, time.reshape(-1,1)])
        e_all = torch.cat([e_all, label.reshape(-1,1)])
#         print(X.shape, y.shape, e.shape)
        if (batch_idx + 1) % 5 == 0:#len(loader):
            
        
#         if (batch_idx + 1) % 10 == 0:
#             print(cls_loss_fn_n.sum(), site_loss_fn_n.sum())
            cls_loss = criterion(X, y, e, model)# + cls_loss_fn_n.sum()# + site_loss_fn_n.sum()
            loss = cls_loss
            cls_loss_value = cls_loss.item()

            cls_train_loss += cls_loss_value

            cls_error = calculate_error(Y_hat, label)
            cls_train_error += cls_error

            # backward pass
            loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()
            all_time += time2.time() - start_time
            print('----------------------------')
            print('cls loss: {:.4f} '.format(cls_loss_value))
            print('batch {}, cls time: {:.4f} '.format(batch_idx, time.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))
            print(f'time cost: {round(all_time, 3)}')
            print('----------------------------')
            X = torch.tensor([]).to(device)
            y = torch.tensor([]).to(device)
            e = torch.tensor([]).to(device)
            logits_all = torch.tensor([]).to(device)
            cls_loss_fn_n = torch.tensor([]).to(device)
            site_loss_fn_n = torch.tensor([]).to(device)
#             cls_loss_fn_n = 0.
#             site_loss_fn_n = 0.
#         else:
#             print(batch_idx, logits.shape, len(loader))
#             cls_logit = logits[:,[0,1]].view(1,2)
#             site_logit = logits[:,2].view(1,1)
#             cls_loss = cls_loss_fn(cls_logit, label)#, time, model)
#             site_loss = site_loss_fn(site_logit.squeeze(-1), time)
            
#             loss = cls_loss * site_loss
            
#             cls_loss_value = cls_loss.item()
#             site_loss_value = site_loss.item()

#             cls_train_loss += cls_loss_value
#             # site_train_loss+=site_loss_value
#             if (batch_idx + 1) % 5 == 0:
#                 print('batch {}, cls loss: {:.4f}, site loss: {:.4f} '.format(batch_idx, cls_loss_value, site_loss_value) +
#                     'label: {}, time: {}, pre time: {}, bag_size: {}'.format(label.item(),  time.item(), float(site_logit),  data.size(0)))
#             # backward pass
#             loss.backward(retain_graph=True)
#             # step
#             optimizer.step()
#             optimizer.zero_grad()
           
        

    # calculate loss and error for epoch
    cls_train_loss /= len(loader)
    cls_train_error /= len(loader)
    train_c = c_index(-X_all, y_all, e_all)
    print('Epoch: {}, cls train_loss: {:.4f}, cls train_error: {:.4f}, cls train_cindex: {:4f}'.format(epoch, cls_train_loss, cls_train_error, train_c))
    text = [str(epoch), '\t', str(train_c), '\n']
    with open('../epoch_cindex.txt', 'a') as f:
        f.writelines(text)
#     for i in range(n_classes):
#         acc, correct, count = cls_logger.get_summary(i)
#         print('class {}: tpr {}, correct {}/{}'.format(i, acc, correct, count))
#         if writer:
#             writer.add_scalar('train/class_{}_tpr'.format(i), acc, epoch)


    if writer:
        writer.add_scalar('train/cls_loss', cls_train_loss, epoch)
        writer.add_scalar('train/cls_error', cls_train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    cls_val_error = 0.
    cls_val_loss = 0.
    
    cls_probs = np.zeros((len(loader), n_classes))
    cls_labels = np.zeros(len(loader))
    criterion = loss_fn[0]
    cls_loss_fn = loss_fn[1]
    site_loss_fn = loss_fn[2]
    print('\n')
    X = torch.tensor([]).to(device)
    y = torch.tensor([]).to(device)
    e = torch.tensor([]).to(device)
    X_all = torch.tensor([]).to(device)
    y_all = torch.tensor([]).to(device)
    e_all = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (data, label, time, gene, cli, _) in enumerate(loader):
            data =  data.to(device)
            label = label.to(device)
            time = time.to(device)
            gene = gene.to(device)
            cli = cli.to(device)

            results_dict = model(data, gene, cli)
            logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
            del results_dict
            
            risk_score = (logits[:,[0,1]].max() * logits[:, 2]).view(1,1)
            X = torch.cat([X, risk_score])
            y = torch.cat([y, time.reshape(-1,1)])
            e = torch.cat([e, label.reshape(-1,1)])
            X_all = torch.cat([X_all, risk_score])
            y_all = torch.cat([y_all, time.reshape(-1,1)])
            e_all = torch.cat([e_all, label.reshape(-1,1)])
#             cls_logger.log(Y_hat, label)
            if (batch_idx + 1) % 10 == 0:
                cls_loss = criterion(X, y, e, model) 
                loss = cls_loss
                cls_loss_value = cls_loss.item()

                cls_probs[batch_idx] = Y_prob.cpu().numpy()
                cls_labels[batch_idx] = label.item()


                cls_val_loss += cls_loss_value
                cls_error = calculate_error(Y_hat, label)
                cls_val_error += cls_error
            

    cls_val_error /= len(loader)
    cls_val_loss /= len(loader)


#     if n_classes == 2:
#         cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
#         cls_aucs = []
#     else:
#         cls_aucs = []
#         binary_labels = label_binarize(cls_labels, classes=[i for i in range(n_classes)])
#         for class_idx in range(n_classes):
#             if class_idx in cls_labels:
#                 fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], cls_probs[:, class_idx])
#                 cls_aucs.append(calc_auc(fpr, tpr))
#             else:
#                 cls_aucs.append(float('nan'))

#         cls_auc = np.nanmean(np.array(cls_aucs))
    
    val_c = c_index(-X_all, y_all, e_all)
    
    if writer:
        writer.add_scalar('val/cls_loss', cls_val_loss, epoch)
#         writer.add_scalar('val/cls_auc', cls_auc, epoch)
        writer.add_scalar('val/cls_error', cls_val_error, epoch)

    print('\nVal Set, cls val_loss: {:.4f}, cls val_error: {:.4f}, cls cindex: {:.4f}'.format(cls_val_loss, cls_val_error, val_c))
#     for i in range(n_classes):
#         acc, correct, count = cls_logger.get_summary(i)
#         print('class {}: tpr {}, correct {}/{}'.format(i, acc, correct, count))
#         if writer:
#             writer.add_scalar('val/class_{}_tpr'.format(i), acc, epoch)
     
    
    if early_stopping:
        assert results_dir
        early_stopping(epoch, cls_val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    cls_test_error = 0.
    cls_test_loss = 0.
    
    all_cls_probs = np.zeros((len(loader), n_classes))
    all_cls_labels = np.zeros(len(loader))
    
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    X_all = torch.tensor([]).to(device)
    y_all = torch.tensor([]).to(device)
    e_all = torch.tensor([]).to(device)
    for batch_idx, (data, label, time, gene, cli, idx) in enumerate(loader):
        data =  data.to(device)
        label = label.to(device)
        time = time.to(device)
        gene = gene.to(device)
        cli = cli.to(device)
        slide_id = slide_ids[idx.item()]
        with torch.no_grad():
            results_dict = model(data, gene, cli)

        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        del results_dict
        
        risk_score = (logits[:,[0,1]].max() * logits[:, 2]).view(1,1)
        X_all = torch.cat([X_all, risk_score])
        y_all = torch.cat([y_all, time.reshape(-1,1)])
        e_all = torch.cat([e_all, label.reshape(-1,1)])
#         cls_logger.log(Y_hat, label)
        cls_probs = Y_prob.cpu().numpy()
        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'logits': risk_score, 'cls_label': label.item(), 'time':time.item()}})
        cls_error = calculate_error(Y_hat, label)
        cls_test_error += cls_error

    cls_test_error /= len(loader)
    patient_results.update({len(loader):{'X': X_all, 'y': y_all, 'e': e_all}})
#     if n_classes == 2:
#         cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])
            
#     else:
#         cls_auc = roc_auc_score(all_cls_labels, all_cls_probs, multi_class='ovr')
    
    test_c = c_index(-X_all, y_all, e_all)
    return patient_results, cls_test_error, test_c, cls_logger

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)