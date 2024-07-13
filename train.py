import os, glob
import numpy as np
from time import time

import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import SEGCRF
import train_utils

from data_utils import MelLabelIntervalCollate, MelLabelIntervalLoader      
from params import ARGS

os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu_idx
device = torch.device("cuda:0" if torch.cuda.is_available() and ARGS.cuda else "cpu")
pkl_foldername = 'train'
def ce_loss(inputs, target, mask):
    ce = F.cross_entropy(inputs, target, reduction='none')
    total_preds = torch.sum(torch.ones_like(mask) * mask)
    return torch.sum(ce) / total_preds

def main(args):

    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    pklList = glob.glob(os.path.join(args.feature_path,
                                     pkl_foldername+'_'+str(args.seg_len)+'_re','*.pkl'))
    
    pklList_train, pklList_val = train_test_split(pklList, test_size=0.03, 
                                                  random_state=2021, 
                                                  shuffle=True)

    train_and_eval(pklList_train, pklList_val)

def train_and_eval(train_list, val_list):

    
    segmenter = SEGCRF(inputdim=ARGS.feature_bins, 
                        numclass=ARGS.num_class, 
                        backbone_type='s4',
                        labelres=int(ARGS.label_res/ARGS.seg_len)).to(device)
    optimizer = optim.Adam(segmenter.parameters(), lr=ARGS.lr)
    
    # dataset define
    train_dataset = MelLabelIntervalLoader(train_list, specAug=ARGS.spec_aug, 
                                            num_class=ARGS.num_class, 
                                            label_res=ARGS.seg_len*1.0/ARGS.label_res,
                                            label_len=ARGS.label_res,
                                            label_map=ARGS.label_map)
    val_dataset = MelLabelIntervalLoader(val_list, num_class=ARGS.num_class, 
                                            label_res=ARGS.seg_len*1.0/ARGS.label_res,
                                            label_len=ARGS.label_res,
                                            label_map=ARGS.label_map)
    collate_fn = MelLabelIntervalCollate(ARGS.num_class)

    # loader define
    train_loader = DataLoader(train_dataset, num_workers=ARGS.num_workers, 
                              shuffle=False, batch_size=ARGS.batch_size,
                              pin_memory=True, drop_last=False, 
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, num_workers=ARGS.num_workers, 
                            shuffle=False, batch_size=ARGS.batch_size,
                            pin_memory=True, drop_last=False, 
                            collate_fn=collate_fn)

    epoch_str = 0
    best_acc = 0
    iteration = 0
    
    # load from checkpoint
    os.makedirs(ARGS.ckpt_path, exist_ok=True)
    os.makedirs(os.path.join(ARGS.ckpt_path, 'ckpts'), exist_ok=True)
    for epoch in range(epoch_str, ARGS.epoch):
        iteration = train(epoch, 
                          iteration, 
                          segmenter, 
                          optimizer, 
                          train_loader, 
                          ARGS.ckpt_path)
        best_acc = evaluate(iteration, 
                            segmenter, 
                            val_loader, 
                            ARGS.ckpt_path, 
                            best_acc)
        torch.save({'backbone_dict':segmenter.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()}, 
                    '{0}/ckpts/model_{1}'.format(ARGS.ckpt_path, epoch))
        
        # Learning rate decay
        if epoch % ARGS.lr_step == 0 and epoch > 0:
            optimizer.param_groups[0]['lr'] *= ARGS.lr_decay

def train(epoch, iteration, model, optimizer, train_loader, ckpt_folder):

    model.train()
    for _, (mels, labels) in enumerate(train_loader):
        
        # Optimization Routine
        optimizer.zero_grad()
        
        orig_time = time()
        mels = mels.to(device, non_blocking=True)

        # Train model
        logp, crf = model(mels, labels)
        loss = (-logp.sum(-1).mean())
        # Compute accuracy
        train_acc = train_utils.accuracy_crf(crf, labels, ARGS.label_res)

        # backwards
        loss.backward()
        optimizer.step()

        # Print training message
        mesg = "Time:{0:.2f}, Epoch:{1}, Iteration:{2}, Loss:{3:.3f}, " \
               "Train Accuracy:{4:.3f}, Learning Rate:{5:.6f}".format(time()-orig_time, 
                epoch, iteration, loss.item(), train_acc, optimizer.param_groups[0]['lr'])
        print(mesg)

        with open(os.path.join(ckpt_folder, 'train_loss.txt'), "a") as f:
            f.write("{0},{1},{2}\n".format(iteration, loss.item(), train_acc))

        iteration += 1

    return iteration

def evaluate(iteration, model, val_loader, ckpt_folder, best_accuracy):
    
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for _, (mels, labels) in enumerate(val_loader):

            mels = mels.to(device, non_blocking=True)
            
            # evaluate model
            logp, crf = model(mels, labels)
            loss = (-logp.sum(-1).mean())
            acc = train_utils.accuracy_crf(crf, labels, ARGS.label_res)
                
            losses.append(loss.item())
            accs.append(acc)

        loss_val = np.mean(np.array(losses))
        acc_val = np.mean(np.array(accs))
        
        with open(os.path.join(ckpt_folder, 'val_loss.txt'), "a") as f:
            f.write("{0},{1},{2}\n".format(iteration, loss_val, acc_val))

    if acc_val > best_accuracy:

        best_accuracy = acc_val
        torch.save({'backbone_dict':model.state_dict()}, 
                   os.path.join(ckpt_folder, 'best_embedding.pt'))
        print('Best model saved!')

    return best_accuracy
                           
if __name__ == "__main__":
    
    main(ARGS)