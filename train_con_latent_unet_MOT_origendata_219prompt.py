from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import os
import os.path as osp
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

MIN_NUM_FRAMES = 25
torch.manual_seed(1)

# Libraries imported - ready to use PyTorch
import warnings
warnings.filterwarnings('ignore')  # Turn off warnings

DEVICE = 'cuda:0'
SEED = 42
SAMPLE_RATE = 44100
batch_size = 4

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

Dmodel = DiffusionModel(
    net_t=UNetV0,
    in_channels=32,
    channels=[128, 256, 512, 512, 1024, 1024],
    factors=[1, 2, 2, 2, 2, 2],
    items=[2, 2, 2, 2, 4, 4],
    attentions=[0, 0, 1, 1, 1, 1],
    attention_heads=8,
    attention_features=64,
    diffusion_t=VDiffusion,
    sampler_t=VSampler,
    use_text_conditioning=True,
    use_embedding_cfg=True,
    embedding_max_length=357,
    embedding_features=219,
    cross_attentions=[1, 1, 1, 1, 1, 1],
)

mot = MOT(
    dim=219,
    depth=6,
    heads=12,
    num_classes=10,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

mot = mot.cuda().to(DEVICE)
Dmodel = Dmodel.to(DEVICE)

torch.cuda.empty_cache()  # Clear memory cache

class VAMDataset(torch.utils.data.Dataset):
    """
    Main class to load video data.
    """
    def __init__(self, video_dir, sampling_rate=44000, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_motion_length = 356
        self.video_files = files_to_list(video_dir)
        self.augment = augment

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        video_file = self.video_files[index]
        smpl_motion = self.load_smpl_motion(video_file)
        filename = video_file.split('/')[-1].split('.')[0]
        audio_path = f'./edge_aistpp/wavs_latent/{filename}.npy'
        audio_latent = self.load_wav_to_torch(audio_path)
        genre_dic = {'gLH': 0, 'gKR': 1, 'gPO': 2, 'gBR': 3, 'gWA': 4, 'gJS': 5, 'gMH': 6, 'gHO': 7, 'gJB': 8, 'gLO': 9}
        genre = filename.split('_')[0]
        genre = genre_dic[genre]
        return audio_latent, smpl_motion, genre, filename

    def load_smpl_motion(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()

    def load_wav_to_torch(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()

va_train_set = VAMDataset(video_dir='edge_aistpp/smpl_motion_219/train_vel_data')
va_train_loader = DataLoader(va_train_set, batch_size=batch_size, num_workers=0, shuffle=True)

from torch.utils.tensorboard import SummaryWriter

date = '0720_0220_mean_(1221cls)_origin_latent!_unet_newmotclsloss_357219prompt'
log_path = f'./log/log_{date}'
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(str(log_path))

def training(Dmodel, mot):
    loss_min = 100
    optimizer = torch.optim.AdamW(Dmodel.parameters(), lr=1e-4, betas=(0.95, 0.99), weight_decay=1e-3, eps=1e-6)
    optM = t.optim.AdamW(mot.parameters(), lr=1e-3, betas=(0.5, 0.9))
    scheduler_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler_optM = torch.optim.lr_scheduler.StepLR(optM, step_size=30, gamma=0.5)
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
        t.backends.cudnn.benchmark = True  # Optimize runtime efficiency

        for i, (a_t, m_t, genre, filename) in tqdm(enumerate(va_train_loader)):
            a_t = a_t.float().to(DEVICE)
            m_t = m_t.float().to(DEVICE)
            genre = genre.to(DEVICE)
            motion, clas = mot(m_t)
            loss_g = loss_genre(clas.squeeze(1), genre.to(dtype=torch.long))
            writer.add_scalar("loss_genre", loss_g.item(), step)
            
            with torch.cuda.amp.autocast():
                loss = Dmodel(a_t.to(DEVICE), text=motion.to(DEVICE), embedding_mask_proba=0.2).to(DEVICE)
                avg_loss += loss.item()
                avg_loss_step += 1
                writer.add_scalar("loss_step", loss.item(), step)

            optimizer.zero_grad()
            optM.zero_grad()
            scaler.scale(0.1*loss).backward(retain_graph=True)
            scaler.step(optimizer)

            scaler.scale(loss_g).backward()
            scaler.step(optM)

            scaler.update()
            progress.update(1)
            progress.set_postfix(
                loss=loss.item(),
                epoch=epoch + i / len(va_train_loader),
            )
            step += 1

        scheduler_optimizer.step()
        scheduler_optM.step()
        writer.add_scalar("optM_learning_rate", optM.defaults['lr'], epoch)
        writer.add_scalar("optimizerD_learning_rate", optimizer.defaults['lr'], epoch)

        torch.cuda.empty_cache()
        writer.close()
        epoch += 1

        Model_path_dir = f'./models/smpl_genre_{date}'
        if not os.path.exists(Model_path_dir):
            os.makedirs(Model_path_dir)

        if epoch % 200 == 0:
            t.save(mot.state_dict(), f"{Model_path_dir}/MOT{epoch}.pt")
            t.save(Dmodel.state_dict(), f'{Model_path_dir}/dancecondation_test{epoch}.pt')

        if epoch % 3 == 0:
            t.save(Dmodel.state_dict(), f'{Model_path_dir}/dancecondation_test_final.pt')
            t.save(mot.state_dict(), f"{Model_path_dir}/MOT_final.pt")

if __name__ == '__main__':
    start_time = time.time
    training(Dmodel, mot)
    end_time = time.time
    using_time = end_time - start_time
    print(f'Operation time: {using_time}')
