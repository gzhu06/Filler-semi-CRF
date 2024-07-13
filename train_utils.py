import torch
import numpy as np

def list_to_interval(num_label, label_list):
    
    i, j = 0, 0
    interval_list = [[] for i in range(num_label)]
    while j < len(label_list):
        if label_list[i] == label_list[j]:
            j += 1
        else:
            interval_list[label_list[i]].append((i, j-1))
            i = j
            j += 1
    interval_list[label_list[i]].append((i, j-1))
    
    return interval_list

def interval_to_list(label_interval, label_resolution):
    
    label_list = [0] * label_resolution
    for i, interval_label in enumerate(label_interval):
        if len(interval_label) == 0:
            continue
        else:
            for interval in interval_label:
                label_list[interval[0]:interval[1]] = [i]*(interval[1]-interval[0])
    return label_list

def accuracy(outputs, target):
    
    # true and pred are both a torch tensor
    correct = 0
    for i, output in enumerate(outputs):

        target_real = target[i]
        output_real = output

        output_real = torch.argmax(output_real, dim=1)
        
        correct += output_real.eq(target_real.data.view_as(output_real)).cpu().sum().numpy()
    return correct * 1.0 / len(target) / len(target[0])

def accuracy_crf(crf, target, label_resolution):
    
    n_class = len(target[0])
    n_batch = len(target)

    # decode predictions:
    with torch.no_grad():
        path = crf.decode()
        lastP = []
        for curP in path:
            if len(curP) == 0:
                lastP.append(0)
            else:
                lastP.append(curP[-1][1])
        intervalsBatch = []
        for idx in range(n_batch):
            curIntervals =  path[idx*n_class: (idx+1)*n_class]
            intervalsBatch.append(curIntervals)
            
    # compute accuracy
    correct = 0
    for i, target_real in enumerate(target):

        target_real_list = interval_to_list(target_real, label_resolution)
        output_real_list = interval_to_list(intervalsBatch[i], label_resolution)
        correct += (np.sum(np.array(output_real_list) == np.array(target_real_list)))
    return correct * 1.0 / len(target) / label_resolution
