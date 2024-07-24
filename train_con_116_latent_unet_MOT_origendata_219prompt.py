'''1210_origin_unet_newmotclsloss,使用在mot中加入了cls监督,是用全部头的,(这当时训练的时候mot的clstoken代码是错误的,!!!!!,
其实我们一直以来用的本来就是clstoken模式,在0108 113_2的代码中我们才尝试把clstoken的模式换成mean pooling的方式了来训练)
与1202_64_219_unet_newmotclsloss相比,
1210用回了原始数据集,因为新裁剪的音频存在保存时有采样率错误的问题

113与110相同,1210使用相同的mot,diffusion的方法换成了latent的方式,从wavs_latent中加载

114继续使用latent diffusion的方法但是将mot换成resnet,做消融实验,效果不好!resnet不如mot
115继续使用latent的方式,加入genre prompt,
0104_origin_latent!_unet_genre_prompt_newmotclsloss:
prompt = 0.1*genre_token + motion  # genre的文字描述与motion编码想结合一起引导diffsion

0105_origin_latent!_unet_genre_prompt_newmotclsloss_0.2:
 prompt = 0.2*genre_token.to(DEVICE) + motion  # genre的文字描述与motion编码想结合一起引导diffsion

0107_origin_latent!_unet_genre_prompt0.01_newmotclsloss:
这里面的0.1*被删掉了,scaler.scale(0.1*loss).backward(retain_graph=True);prompt = 0.01*genre_token.to(DEVICE) + motion

113_2,1210使用相同的mot,diffusion的方法换成了latent的方式,从wavs_latent中加载,113使用的mot的分类头进行监督,113_2使用mot输出的全部的平均池化

0129_mean_(1221cls)_origin_latent!_unet_newmotclsloss_2_0126
属于当前最新完整的模型,他们的u-net的attention的输入大小为64,768. embedding_max_length=64, embedding_features=768,

   0220 :
   设法减少unet的输入,更改mot的输出大小,不在接入self.model来改变输出大小

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
import time
from random import choice
import random
# Import PyTorch libraries
import torch
import torch as t
from a__unet import vqEncoder_low, motion_encoder, vqEncoder_high , MOT


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

# from transformers import AutoModel
# diffAE_model = AutoModel.from_pretrained(pretrained_model_name_or_path='./edge_aistpp/DMAE1d-ATC32-v3/',trust_remote_code=True)
# diffAE_model.eval()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)

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

mot = mot.cuda().to(DEVICE)
Dmodel = Dmodel.to(DEVICE)


torch.cuda.empty_cache()  # 清空显存缓冲区

# MODELPATH = './models/smpl_genre_0516/dancecondation_test0516.pth'
# load_path = './models/smpl_genre_0516'
# checkpoint = torch.load(MODELPATH)
# Dmodel.load_state_dict(checkpoint['model_state_dict'])  # 载入之前训练的模型



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

        # List = [0, 1]
        # randomx = choice(List)  # 随机选择０１秒起点
        # motion_random_start = int(randomx) * 60
        # music_random_start = int(randomx) * sampling_rate
        motion_random_start =0

        # random start music and motion
        smpl_motion = smpl_motion[(motion_random_start):(motion_random_start + self.segment_motion_length), :]
        # audio = audio[(music_random_start):(music_random_start + self.segment_wav_length)]


        # read genre
        genre_dic = {'gLH': 0, 'gKR': 1, 'gPO': 2, 'gBR': 3, 'gWA': 4, 'gJS': 5, 'gMH': 6, 'gHO': 7, 'gJB': 8, 'gLO': 9}
        genre = filename.split('_')[0]
        genre = genre_dic[genre]

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

va_train_set = VAMDataset(video_dir='edge_aistpp/smpl_motion_219/train_vel_data')
va_train_loader = DataLoader(va_train_set, batch_size = batch_size, num_workers=0, shuffle=True)

#*********************加载之前训练的模型************************************#
# Dmodel.load_state_dict(torch.load("./models/smpl_genre_1221_origin_latent!_unet_newmotclsloss/dancecondation_test_final.pt"))
# mot.load_state_dict(torch.load("./models/smpl_genre_1221_origin_latent!_unet_newmotclsloss/MOT_final.pt"))
#**********************************************************#


# tensorboard
from torch.utils.tensorboard import SummaryWriter

# date = '0220_mean_(1221cls)_origin_latent!_unet_newmotclsloss_357219prompt'  # 开始训练的日期

date = '0720_0220_mean_(1221cls)_origin_latent!_unet_newmotclsloss_357219prompt'  # 开始训练的日期

log_path = f'./log/log_{date}'  # 创建保存日志log的文件夹
# log_path=f'./log/log_0606'
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(str(log_path))


####训练模型
def training(Dmodel,mot):
    loss_min = 100 #用于保存最好eval_loss模型
    # Use an "Adam" optimizer to adjust weights
    optimizer = torch.optim.AdamW(Dmodel.parameters(), lr=1e-4 ,betas=(0.95, 0.99),weight_decay = 1e-3,eps = 1e-6)
    
    mot.parameters()
    optM = t.optim.AdamW(mot.parameters(), lr=1e-3, betas=(0.5, 0.9))  # motion video genre

    scheduler_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, last_epoch=-1)  # 设置逐步减小的变化学习率
    scheduler_optM = torch.optim.lr_scheduler.StepLR(optM, step_size=30, gamma=0.5 )  # 设置逐步减小的变化学习率

    epoch = 0
    step = 0
    print('Training on', DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    loss_genre = torch.nn.CrossEntropyLoss(reduction="mean")

    while epoch < 2000:
        avg_loss = 0
        avg_loss_step = 0
        Dmodel.train()
        progress = tqdm(va_train_loader)
        t.backends.cudnn.benchmark = True  # 优化运行效率,网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用

        for i, (a_t, m_t, genre, filename) in tqdm(enumerate(va_train_loader)):
            # get video, audio and beat data
            
            a_t = a_t.float().to(DEVICE)  # torch.Size([2, 1, 262144])

            m_t = m_t.float().to(DEVICE)  # m_t:torch.Size([2, 356, 219])

            # genre = torch.ToTensor(genre)
            genre = genre.to(DEVICE)  # torch.Size([2, 1]) tensor(int)

            # get output from encoder
            motion , clas = mot(m_t)  # mx = mencoder(m_t):torch.Size([2, 1, 1024])

            # print(genre.shape)
            # print(clas.shape)
            # print(clas.squeeze(1))
            # print(genre.to(dtype=torch.long))

            loss_g = loss_genre(clas.squeeze(1),genre.to(dtype=torch.long))
            writer.add_scalar("loss_genre", loss_g.item(), step)
           
            
            with torch.cuda.amp.autocast():
                loss = Dmodel(a_t.to(DEVICE), text=motion.to(DEVICE),  # Text conditioning, one element per batch
                              embedding_mask_proba=0.2
                              # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
                              ).to(DEVICE)
                # print(loss.size())
                avg_loss += loss.item()
                avg_loss_step += 1
                # tensorboard
                writer.add_scalar("loss_step", loss.item(), step)

            # sumloss=torch.zeros(loss.size())
            # print(loss.size())
            # sumloss=sumloss+loss


            optimizer.zero_grad()
            optM.zero_grad()
            scaler.scale(0.1*loss).backward(retain_graph=True)
            scaler.step(optimizer)

            scaler.scale(loss_g).backward()
            
            scaler.step(optM)



            # print([x.grad for x in optimizer.param_groups[0]['params']])# 查看优化器的参数
            scaler.update()
            progress.update(1)
            progress.set_postfix(
                loss=loss.item(),
                epoch=epoch + i / len(va_train_loader),
            )
                

            if step % 20 == 0:
                print(f'epoch:{epoch}  step:{step}  loss: {loss}   avg_loss(100){avg_loss / avg_loss_step}')
                writer.add_scalar("avg_loss(100)_epoch", avg_loss / avg_loss_step, step)
                avg_loss = 0
                avg_loss_step = 0

            step += 1

        scheduler_optimizer.step()  ###设置逐步减小的变化学习率
        scheduler_optM.step()  ###设置逐步减小的变化学习率
        writer.add_scalar("optM_learning_rate", optM.defaults['lr'], epoch)
        writer.add_scalar("optimizerD_learning_rate", optimizer.defaults['lr'], epoch)



        # if epoch % 10 == 0:  #验证集
        #     Dmodel.eval()
        #     mot.eval()
        #     sun_loss_eval = 0
        #     for i, (a_t, m_t, genre, filename) in tqdm(enumerate(va_eval_loader)):
        #         # get video, audio and beat data

        #         a_t = a_t.float().to(DEVICE)  # torch.Size([2, 1, 262144])
        #         m_t = m_t.float().to(DEVICE)  # m_t:torch.Size([2, 356, 219])
        #         genre = genre.to(DEVICE)  # torch.Size([2, 1]) tensor(int)

        #         # get output from encoder
        #         motion, clas = mot(m_t)  # mx = mencoder(m_t):torch.Size([2, 1, 1024])
               
        #         with torch.cuda.amp.autocast():
        #             loss_eval = Dmodel(a_t.to(DEVICE), text=motion.to(DEVICE),  # Text conditioning, one element per batch
        #                           embedding_mask_proba=0.2
        #                           # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
        #                           ).to(DEVICE)
        #             writer.add_scalar("eval_step", loss_eval.item(), step+i) #输出验证集的没有loss
        #             sun_loss_eval = sun_loss_eval + loss_eval.item()
        #     avg_loss_eval = sun_loss_eval/24  #验证集总共24个clip
        #     writer.add_scalar("avg_eval_step", avg_loss_eval , epoch)   #打印验证集的平均loss
        
        torch.cuda.empty_cache()  # 清空显存缓冲区 
        writer.close()
        epoch += 1
        


        # model save###########################################################
       
        Model_path_dir = f'./models/smpl_genre_{date}'  # 创建保存模型的文件夹
        if not os.path.exists(Model_path_dir):
            os.makedirs(Model_path_dir)

        if epoch % 200 == 0:
            # t.save({
            #     'epoch': epoch,
            #     'model_state_dict': Dmodel.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, f'{Model_path_dir}/dancecondation_test{epoch}.pth')
            t.save(mot.state_dict(), f"{Model_path_dir}/MOT{epoch}.pt")
            t.save(Dmodel.state_dict(), f'{Model_path_dir}/dancecondation_test{epoch}.pt')
            # t.save(optM.state_dict(), f"{Model_path_dir}/optM{epoch}.pt")

           
        # if avg_loss_eval < loss_min:
        #     loss_min=avg_loss_eval
        #     print(f"\n---------------------------------save best model epoch:{epoch}--------------------------------\n")
      
        #     t.save(Dmodel.state_dict(), f'{Model_path_dir}/dancecondation_test{epoch}_best.pt')
        #     t.save(mot.state_dict(), f"{Model_path_dir}/MOT{epoch}_best.pt")
        #     # torch.save(optM.state_dict(), f"{Model_path_dir}/optM_best.pt")
            
        if epoch % 3 == 0:
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': Dmodel.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, f'{Model_path_dir}/dancecondation_test_final.pth')
            t.save(Dmodel.state_dict(), f'{Model_path_dir}/dancecondation_test_final.pt')
            t.save(mot.state_dict(), f"{Model_path_dir}/MOT_final.pt")
            # torch.save(optM.state_dict(), f"{Model_path_dir}/optM_final.pt")


if __name__ == '__main__':
    
    start_time=time.time
    training(Dmodel,mot)
    end_time = time.time
    using_time= end_time - start_time
    print(f'Operation time:{using_time}')