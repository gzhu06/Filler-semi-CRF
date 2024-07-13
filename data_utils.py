import random
import numpy as np
import pickle5 as pickle

import torch
import torch.utils.data
import torchaudio.transforms as T

def clipped_feature(x, num_frames, bias=None):
    
    if bias is None:
        bias = np.random.randint(0, x.shape[-1] - num_frames)
    
    clipped_x = x[:, bias: num_frames + bias]

    return clipped_x
    
class MelLabelIntervalLoader(torch.utils.data.Dataset):
    """
        featurePaths: a list of audio feature files
    """
    def __init__(self, featurePaths, 
                 specAug=False, 
                 num_class=5,
                 label_res=10,
                 label_len=20,
                 label_map=None):
                 
        self.featurePaths = featurePaths
        self.Fmasking = T.FrequencyMasking(freq_mask_param=10)    # vertical masking, time
        self.Tmasking = T.TimeMasking(time_mask_param=50)         # horizontal masking, frequency
        self.specAug = specAug
        self.label_res = label_res
        self.label_map = label_map
        self.num_class = num_class
        self.label_len = label_len
        random.seed(2021)
        random.shuffle(self.featurePaths)

    def get_mel_label_pair(self, featureFile):

        with open(featureFile, 'rb') as handle:
            featureData = pickle.load(handle)

        # separate log mels and labels
        log_mel, labels = featureData['feature'][0], featureData['label']
        
        # acoustic feature
        log_mel = torch.from_numpy(log_mel)
        if self.specAug:
            randMask = random.uniform(0, 1)
            if randMask > 0.5:
                log_mel = self.Tmasking(log_mel)
            else:
                log_mel = self.Fmasking(log_mel)
            
        # label
        intervals = [[] for i in range(self.num_class)]
        for label_event in labels:
            time_s, time_e = label_event[0], label_event[1]
            event_label = label_event[-1]
            if event_label == 'uh' or event_label == 'um':
                event_label = 'Filler'
            time_s_int = min(int(np.round(time_s/self.label_res)), self.label_len-1)
            time_e_int = min(int(np.round(time_e/self.label_res)), self.label_len-1)
            if time_s_int > time_e_int:
                continue
            intervals[self.label_map[event_label]].append((time_s_int, time_e_int))
        return (log_mel, intervals)

    def __getitem__(self, index):
        return self.get_mel_label_pair(self.featurePaths[index])

    def __len__(self):
        return len(self.featurePaths)

class MelLabelIntervalCollate():

    def __init__(self, num_class, *params):
        self.params = params
        self.num_class = num_class

    def __call__(self, batch):
        """
            Collate's training batch from mel-spectrogram and interval labels
        PARAMS
        ------
        batch: [log mels, filler labels]
        """
        
        num_mels = batch[0][0].size(0)
        size_seg = batch[0][0].size(1)
        
        mel_padded = torch.FloatTensor(len(batch),  num_mels, size_seg)
        mel_padded.zero_()
        
        label_padded = []
        for i, x in enumerate(batch):
            
            mel, label_interval = x[0], x[1]
            
            mel_padded[i, :mel.shape[0], :] = mel
            label_padded.append(label_interval)
        return mel_padded, label_padded
