'''1210_origin_unet_newmotclsloss,使用在mot中加入了cls监督,是用全部头的,
与1202_64_219_unet_newmotclsloss相比,
1210用回了原始数据集,因为新裁剪的音频存在保存时有采样率错误的问题
113与110相同,1210使用相同的mot,diffusion的方法换成了latent的方式,从wavs_latent中加载

smpl_genre_0129_mean_(1221cls)_origin_latent!_unet_newmotclsloss_2_0126  截止到0202电脑修好之后再重新训练出的模型跑了5000个epoch
'''

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import os
import os.path as osp

##os.environ['PYOPENGL_PLATFORM'] = 'egl'

# import cv2
# import time
import torch
# import joblib
# import shutil
# import colorsys
# import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader


from scipy.spatial.transform import Rotation as R

MIN_NUM_FRAMES = 25
# random.seed(1)
torch.manual_seed(1)
# np.random.seed(1)

#####
# from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import numpy as np
import gc
# import os for system operations
import os

from random import choice
import random
# Import PyTorch libraries
import torch
import torch as t
from a__unet import   MOT
# from noisereduce import reduce_noise #降噪

import torchaudio

from librosa.core import load
from librosa.util import normalize

# import warnings to hide the unnessairy warniings
import warnings

warnings.filterwarnings('ignore')  # 关闭警告

print("Libraries imported - ready to use PyTorch", torch.__version__)

# os.environ['CUDA_VISIBLE_DEVICES']= '0'  #使代码只能看到0号显卡
DEVICE = 'cuda:0'
SEED = 42
SAMPLE_RATE = 44100
batch_size = 4

from transformers import AutoModel
diffAE_model = AutoModel.from_pretrained(pretrained_model_name_or_path='./edge_aistpp/DMAE1d-ATC32-v3/',trust_remote_code=True)
diffAE_model.to(DEVICE)



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)

# Dmodel = DiffusionModel(
#     net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
#     in_channels=32, # U-Net: number of input/output (audio) channels
#     channels=[128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
#     factors=[1, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
#     items=[2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
#     attentions=[0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
#     attention_heads=8, # U-Net: number of attention heads per attention item
#     attention_features=64, # U-Net: number of attention features per attention item
#     diffusion_t=VDiffusion, # The diffusion method used
#     sampler_t=VSampler, # The diffusion sampler used
#     use_text_conditioning=True,
#     use_embedding_cfg=True,
#     embedding_max_length=64,
#     embedding_features=768,
#     cross_attentions=[1, 1, 1, 1, 1, 1],
# )

Dmodel = DiffusionModel(
    net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
    in_channels=32, # U-Net: number of input/output (audio) channels
    channels=[128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=8, # U-Net: number of attention heads per attention item
    attention_features=64, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
    use_text_conditioning=True,
    use_embedding_cfg=True,
    embedding_max_length=357, #64
    embedding_features=219,  #768
    cross_attentions=[1, 1, 1, 1, 1, 1],
)



mot = MOT(
    
    dim = 219,
    depth = 6,
    heads = 12,
    num_classes = 10,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)

mot = mot.to(DEVICE)
Dmodel = Dmodel.to(DEVICE)

torch.cuda.empty_cache()  # 清空显存缓冲区


######
##读入视频
def files_to_list(dirname):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    files = []
    filenames = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        files.append(f'{dirname}/{filename}')
    files = [f.rstrip() for f in files]
    return files


class VAMDataset(torch.utils.data.Dataset):
    """
    This is the main class to load video data.
    """

    def __init__(self, video_dir, sampling_rate=44000, augment=True):

        self.sampling_rate = sampling_rate
        # self.segment_wav_length = 262144
        self.segment_motion_length = 356
        self.video_files = files_to_list(video_dir)

        # self.audio_files = files_to_list(audio_files)
        # self.audio_files = [Path(audio_files).parent / x for x in self.audio_files]
        self.augment = augment
        # self.video_files = files_to_list(video_files)
        # self.video_files = [Path(video_files).parent / x for x in self.video_files]
        # self.motion_files = files_to_list(motion_files)
        # self.motion_files = [Path(motion_files).parent / x for x in self.motion_files]
        # self.genre_files = files_to_list(genre_files)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        # Read motion
        video_file = self.video_files[index]
        smpl_motion = self.load_smpl_motion(video_file)
        # smpl_motion = smpl_motion[0:178, :]  # 裁剪一下motion的长度0-356 smpl_motion (356, 219)
     
        # Read filename
        filename = video_file.split('/')[-1].split('.')[0]  # 获得文件名

        # Read audio
        audio_path = f'./edge_aistpp/wavs_latent/{filename}.npy'  # 分离出的音频的存放地址
        audio_latent = self.load_wav_to_torch(audio_path)

        # Read filename
        filename = video_file.split('/')[-1].split('.')[0]  # 获得文件名

        # Read genre
        motion_random_start =0

        # random start music and motion
        smpl_motion = smpl_motion[(motion_random_start):(motion_random_start + self.segment_motion_length), :]
        # audio = audio[(music_random_start):(music_random_start + self.segment_wav_length)]

        # read genre
        genre_dic = {'gLH': 0, 'gKR': 1, 'gPO': 2, 'gBR': 3, 'gWA': 4, 'gJS': 5, 'gMH': 6, 'gHO': 7, 'gJB': 8, 'gLO': 9}
        genre = filename.split('_')[0]
        genre = genre_dic[genre]

        print(filename)
        return audio_latent, smpl_motion ,genre, filename


    def load_smpl_motion(self, full_path):
        """ Prepare input video (images) """

        data = np.load(full_path)
        return torch.from_numpy(data).float()

    def load_genre_token(self, genre):
        """ Prepare input video (images) """

        data = np.load(f'edge_aistpp/genre_token/{genre}.npy')
        return torch.from_numpy(data).float()

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data = np.load(full_path)
        return torch.from_numpy(data).float()


# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

va_test_set = VAMDataset(video_dir='edge_aistpp/smpl_motion_219/test_data')
va_test_loader = DataLoader(va_test_set, batch_size =1, num_workers=0, shuffle=True)

#*********************Load the previously trained model************************************#

Dmodel.load_state_dict(torch.load("./models/smpl_genre_0720_0220_mean_(1221cls)_origin_latent!_unet_newmotclsloss_357219prompt/dancecondation_test_final.pt"))
mot.load_state_dict(torch.load("./models/smpl_genre_0720_0220_mean_(1221cls)_origin_latent!_unet_newmotclsloss_357219prompt/MOT_final.pt"))

#***************************Where to save the generated music*******************************#

savedir='./Generate/0720_0220_mean(1221cls)_origin_latent_unet_newmot357219prompt'
os.makedirs(savedir,exist_ok=True)


####训练模型
def generate(Dmodel,mot,diffAE_model):

    print('Training on', DEVICE)

    Dmodel.eval()
    mot.eval()
    diffAE_model.eval

    for i, (a_t, m_t, genre, filename) in tqdm(enumerate(va_test_loader)):
        # get video, audio and beat data

        a_t = a_t.float().to(DEVICE)  # torch.Size([1,32,256])audio latent

        m_t = m_t.float().to(DEVICE)  # m_t:torch.Size([1, 356, 219])

        # genre = torch.ToTensor(genre)
        genre = genre.to(DEVICE)  # torch.Size([2, 1]) tensor(int)tensor([9])

        # get output from encoder
        motion , clas = mot(m_t)  # motion(1,64,768) clastensor.size([1,10])

        # print(genre.shape)
        # print(clas.shape)
        # print(clas.squeeze(1))
        # print(genre.to(dtype=torch.long))


        noise = torch.randn(1, 32, 256, device=DEVICE)  # 生成音乐出！！
        with torch.cuda.amp.autocast():
            audio_latent = Dmodel.sample(noise, num_steps=50, text=motion,
                                  embedding_scale=9, #embedding_scale=,
                                  # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
                                  )
            # sample = diffAE_model.decode(audio_latent, num_steps=10)
            sample = diffAE_model.decode(audio_latent, num_steps=50)

            sample = sample.squeeze(0).cpu()
            # 设置去噪参数
            noise_sample =sample[:,:30000]  # 取前一段音频作为噪声样本，根据需要调整
            n_grad_freq = 2  # 频域去噪参数，根据需要调整
            n_grad_time = 4  # 时间域去噪参数，根据需要调整
            n_fft = 2048  # FFT 窗口大小，根据需要调整
            
            torchaudio.save(f'{savedir}/test_generated_latent_{filename}_884_embedding_scale4_44100.wav',
                            sample,
                            SAMPLE_RATE)
            print(filename)

            # 打印模型参数量
            print(f'Dmodel Total number of parameters: {count_parameters(Dmodel)}')

            del sample


if __name__ == '__main__':
    


    generate(Dmodel,mot,diffAE_model)
    # print(f'mot Total number of parameters: {count_parameters(mot)}')
    # print(f'diffAE_model Total number of parameters: {count_parameters(diffAE_model)}')