# hyperparameters:
SEGMENT_LEN = 2 # segment length for training the filler segmenter
LABEL_RESOLUTION = 40 # number of labels within segment length
PROCESS_WIN = 2
BACKBONE = 's4'
GPU_IDX = '0'
LR = 1e-3
SKIP_SCORE = False
FEATURE_TYPE = 'logmel'
SEGMENT_PKL_PATH = '/home/gzhu/'

# audio signal processing parameters
SAMPLE_RATE = 16000
MEL_WINDOW_LENGTH = 25  # window size for logmel, in ms
MEL_HOP_LENGTH = 10  # hop size for logmel, in ms
MEL_BINS = 64
FFT_SIZE = 1024
FMIN = 0  
FMAX = 8000
MIN_DURATION = 150
MAX_DURATION = 2000

# classification paramters
LABELS = ['Words', 'Filler', 'Laughter', 'Music', 'Breath']
NUM_CLASS = len(LABELS)
CLASS_MAP = {}
LABEL_MAP = {}
for i, label in enumerate(LABELS):
    CLASS_MAP[i] = label
    LABEL_MAP[label] = i

# training parameters
CUDA = True  # True for GPU
SPEC_AUG = True  # specAug: Fmasking and Tmasking
BATCH_SIZE = 256
NUM_WORKERS = 4
EPOCH = 30
LR_STEP = 10
LR_DECAY = 0.6

# path parameters
import os
PF_ROOT = '/home/gzhu/pf_root'
SEGMENT_PATH = "/storage/ge/filler/filler_segmenter/" + str(SEGMENT_LEN)
EXP_NAME = FEATURE_TYPE + '_' + BACKBONE + '_' + str(SEGMENT_LEN) + '_res_' + str(LABEL_RESOLUTION)
WAV2VEC_PATH = "/home/gzhu/wav2vec1.pt"

# other parameters
if FEATURE_TYPE == 'logmel':
    FEATURE_BINS = 64     # mel: 64, 
    PKL_PATH = os.path.join(SEGMENT_PKL_PATH, 'logmel_' + str(SEGMENT_LEN))
elif 'wav2vec' in FEATURE_TYPE:
    FEATURE_BINS = 512     # wav2vec: 512
    PKL_PATH = os.path.join(SEGMENT_PKL_PATH, 'wav2vec_' + str(SEGMENT_LEN))
    
from types import SimpleNamespace
skip_score_module = '_skip' if SKIP_SCORE and 'crf' in BACKBONE else ''
model_ckpt_path = os.path.join('./exps/', EXP_NAME + skip_score_module)
ARGS = SimpleNamespace(# path related parameters
                       feature_path=PKL_PATH, 
                       audio_path=SEGMENT_PATH,
                       ckpt_path=model_ckpt_path,
                       pf_root=PF_ROOT,
                       wav2vec_path=WAV2VEC_PATH,
                       # training related parameters
                       num_class=NUM_CLASS, 
                       epoch=EPOCH, lr=LR,
                       batch_size=BATCH_SIZE, 
                       skip_score=SKIP_SCORE, 
                       lr_decay=LR_DECAY, 
                       spec_aug=SPEC_AUG, 
                       num_workers=NUM_WORKERS, 
                       lr_step=LR_STEP, 
                       labels=LABELS, 
                       gpu_idx=GPU_IDX, cuda=CUDA,
                       label_res=LABEL_RESOLUTION, 
                       label_map=LABEL_MAP, 
                       classmap=CLASS_MAP,
                       # audio feature related parameters
                       feature_type=FEATURE_TYPE,
                       feature_bins=FEATURE_BINS, 
                       sample_rate=SAMPLE_RATE, 
                       mel_window_length=MEL_WINDOW_LENGTH, 
                       mel_bins=MEL_BINS, 
                       mel_hop_length=MEL_HOP_LENGTH, 
                       fft_size=FFT_SIZE, 
                       fmin=FMIN, fmax=FMAX, 
                       # post processing related parameters
                       seg_len=SEGMENT_LEN, 
                       process_win=PROCESS_WIN,
                       min_duration=MIN_DURATION,
                       max_duration=MAX_DURATION)