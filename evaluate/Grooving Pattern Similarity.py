'''
Grooving Pattern Similarity (GS) 是用于衡量音乐节奏感的评价指标。
如果一段音乐有明显的节奏感，GS 分数将会较高。GS 的具体计算方法可能涉及到音乐节奏模式的提取和比较。

1221_origin_latent_unet_newmotclsloss_2:
mean Grooving Pattern Similarity (GS): 0.13315636507774653

audio_result_ait_low_0124_merge:
mean Grooving Pattern Similarity (GS): 0.17727219834923744
audio_result_ait_low_0124_merge_2:
mean Grooving Pattern Similarity (GS): 0.1750158816576004

0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss:
mean Grooving Pattern Similarity (GS): 0.13693819791078568

0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2:
mean Grooving Pattern Similarity (GS): 0.1193891704082489#hop_size=512
mean Grooving Pattern Similarity (GS): 0.094443473033607#hop_size=128
mean Grooving Pattern Similarity (GS): 0.16071346625685692#hop_size=1024

0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2:

GT (edge_aistpp/split_music/test_music):
0.15812707915902138 #hop_size=1024
0.12836328148841858 #hop_size=512


'''

import librosa
import numpy as np
import os

def grooving_pattern_similarity(audio_file, hop_size=512):
    # 加载音频文件
    y, sr = librosa.load(audio_file)

    # 提取音频的节奏特征（例如：onset envelope）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_size)
    # onset_env = librosa.onset.onset_strength(y=y, sr=sr) #hop_length不输入默认512

    # 对节奏特征进行归一化
    onset_env /= onset_env.max()

    # 计算 GS 分数（示例中简单地取节奏特征的均值）
    gs_score = np.mean(onset_env)

    return gs_score

# 示例用法

#########################################################################音频文件路径
# path = '../Generate/1221_origin_latent_unet_newmotclsloss_2/wav/'
# path = '../Generate/audio_result_ait_low_0124_merge/wav'
# path = '../Generate/audio_result_ait_low_0124_merge_2/wav'
# path = '../Generate/0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss/wav'
# path = '../Generate/0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2/wav'
path = '../edge_aistpp/split_music/test_music' # 运行需要注释掉下面的filename = filename.split("'")[1]



#########################################################################


fileList = os.listdir(path)
seq_names =[]
for filename in fileList:
    filename = filename.split("'")[1]

    seq_names.append(filename)


i=0
gs_result_sum=0
for seq_name in seq_names:

    # Produce a batch of log mel spectrogram examples.
    generate_audio_file = f'{path}/{fileList[i]}'
    # wav_file_r = f'wavs_clip_norepeat/{seq_name}.wav'  #
    # GT_audio_file = f'../edge_aistpp/wavs/{seq_name}.wav'
    i=i+1

    gs_result = grooving_pattern_similarity(generate_audio_file)

    print(f"Grooving Pattern Similarity (GS): {gs_result}")
    gs_result_sum+=gs_result

print(f"mean Grooving Pattern Similarity (GS): {gs_result_sum/len(seq_names)}")