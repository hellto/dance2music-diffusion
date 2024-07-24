'''测试audio-diffusion的幅度普压缩怎么用 latent '''
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import numpy as np
import gc

# import os for system operations
import os
# import random for get random values/choices
import random
from random import choice
# Import PyTorch libraries
import torch
import torch as t


from librosa.core import load
from librosa.util import normalize
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# from torchsummary import summary
# from torchview import draw_graph

from pathlib import Path
import torchaudio

#

print("Libraries imported - ready to use PyTorch", torch.__version__)

# import tqdm to show a smart progress meter
from tqdm.notebook import trange, tqdm

# import warnings to hide the unnessairy warniings
import warnings

warnings.filterwarnings('ignore') #关闭警告
from transformers import AutoModel
diffAE_model = AutoModel.from_pretrained(pretrained_model_name_or_path='./DMAE1d-ATC32-v3/',trust_remote_code=True)
diffAE_model.eval()


DEVICE = 'cuda:0'
SEED = 42
SAMPLE_RATE = 44100
batch_size = 1  #重新训练前注意检查　学习率、读取和保存模型、保存log　的位置

#是否是接续训练
# Resume = True
Resume = False



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)



def extract_music_latent(diffAE_model):
    # Use an "Adam" optimizer to adjust weights
    files = []
    path_ = './wavs/'
    filenames = os.listdir(path_)  # 写文件夹地址
    for filename in tqdm(filenames):

    # get  audio 
        path = f"{path_}{filename}"  # 待语音的语音
        path_2 = "./wavs_latent/"  # 分割后语音放置的文件夹
        audio_wave, sampling_rate = load(path, sr=SAMPLE_RATE, mono=False) #设置保持双声道mono=False
        audio_wave = torch.from_numpy(audio_wave).unsqueeze(0).to(DEVICE)

        audio_wave = audio_wave[:,:,:262144]  #torch.Size([2, 1, 262144])

        print(audio_wave.size())
        #压缩64x
        audio_latent = diffAE_model.encode(audio_wave)
        print(audio_latent.size())
        # NUM_steps=30
        # sample = diffAE_model.decode(audio_latent[: ,: , :],num_steps=NUM_steps)
        # print(sample.size())
        music_name = filename.split('.')[0]
        print(music_name)
        np.save(f'{path_2}{music_name}.npy',audio_latent.squeeze(0).cpu().detach().numpy())


        
        
        # torchaudio.save(f'./Generate/latent_test/normalize_decode{filename}{NUM_steps}.wav', sample.squeeze(0).cpu(),
        #                 SAMPLE_RATE)
        # torchaudio.save(f'./Generate/latent_test/normalize_o_encode{filename}.wav', audio_wave.squeeze(0).cpu(),
        #                 SAMPLE_RATE)
    





if __name__ == '__main__':
    extract_music_latent(diffAE_model.to(DEVICE))




