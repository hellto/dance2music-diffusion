import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from a__unet import MOT
from transformers import AutoModel
import torchaudio
import warnings

# Constants
DEVICE = 'cuda:0'
SEED = 42
SAMPLE_RATE = 44100
BATCH_SIZE = 4
SAVEDIR = './Generate/0129_mean_(1221cls)_origin_latent_unet_newmot_mean_clsloss'
LOG_DATE = '0720_0220_mean_(1221cls)_origin_latent!_unet_newmotclsloss_357219prompt'

warnings.filterwarnings('ignore')
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Create necessary directories
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

# Initialize models
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

mot = mot.to(DEVICE)
Dmodel = Dmodel.to(DEVICE)
diffAE_model = AutoModel.from_pretrained(pretrained_model_name_or_path='./edge_aistpp/DMAE1d-ATC32-v3/', trust_remote_code=True)
diffAE_model.to(DEVICE)

torch.cuda.empty_cache()


# Utility functions
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)


def files_to_list(dirname):
    filenames = os.listdir(dirname)
    return [f'{dirname}/{filename}' for filename in filenames]


# Dataset class
class VAMDataset(torch.utils.data.Dataset):
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
        motion_random_start = 0
        smpl_motion = smpl_motion[(motion_random_start):(motion_random_start + self.segment_motion_length), :]
        genre_dic = {'gLH': 0, 'gKR': 1, 'gPO': 2, 'gBR': 3, 'gWA': 4, 'gJS': 5, 'gMH': 6, 'gHO': 7, 'gJB': 8, 'gLO': 9}
        genre = genre_dic[filename.split('_')[0]]
        return audio_latent, smpl_motion, genre, filename

    def load_smpl_motion(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()

    def load_wav_to_torch(self, full_path):
        data = np.load(full_path)
        return torch.from_numpy(data).float()


# Load test set
va_test_set = VAMDataset(video_dir='edge_aistpp/smpl_motion_219_complete/test_data')
va_test_loader = DataLoader(va_test_set, batch_size=1, num_workers=0, shuffle=True)

# Load pre-trained models
Dmodel.load_state_dict(torch.load("./models/smpl_genre_0129_mean_(1221cls)_origin_latent!_unet_newmotclsloss_2_0126/dancecondation_test_final.pt"))
mot.load_state_dict(torch.load("./models/smpl_genre_0129_mean_(1221cls)_origin_latent!_unet_newmotclsloss_2_0126/MOT_final.pt"))


# Generate function
def generate(Dmodel, mot, diffAE_model):
    Dmodel.eval()
    mot.eval()
    diffAE_model.eval()

    for i, (a_t, m_t, genre, filename) in tqdm(enumerate(va_test_loader)):
        a_t = a_t.float().to(DEVICE)
        m_t = m_t.float().to(DEVICE)
        genre = genre.to(DEVICE)

        motion, clas = mot(m_t)

        noise = torch.randn(1, 32, 256, device=DEVICE)
        with torch.cuda.amp.autocast():
            audio_latent = Dmodel.sample(noise, num_steps=50, text=motion, embedding_scale=9)
            sample = diffAE_model.decode(audio_latent, num_steps=50).squeeze(0).cpu()
            torchaudio.save(f'{SAVEDIR}/test_generated_latent_{filename}_44100.wav', sample, SAMPLE_RATE)
            del sample


if __name__ == '__main__':
    generate(Dmodel, mot, diffAE_model)
