'''
Pitch Class Histogram Entropy (PHE) 用于评估音乐的音调质量。
如果音乐作品的主音清晰可辨，那么 PHE 分数将较高。在计算 PHE 时，通常需要计算音高直方图并根据直方图计算熵。
我们使用 librosa 库加载音频文件，提取音高信息，然后计算音高直方图，最后计算 Pitch Class Histogram Entropy。
与前述的 Grooving Pattern Similarity 示例类似，你可能需要根据具体的定义和需求进行更复杂的特征提取和分析。

audio_result_ait_low_0124_merge_2:
Pitch Class Histogram Entropy sum (PHE): 3.500529008636563
audio_result_ait_low_0124_merge:
Pitch Class Histogram Entropy sum (PHE): 3.523573735604937


1221_origin_latent_unet_newmotclsloss_2:
Pitch Class Histogram Entropy sum (PHE): 3.6048268225032363

0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss:
Pitch Class Histogram Entropy sum (PHE): 3.6569651599571373

0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2:
Pitch Class Histogram Entropy sum (PHE): 3.531316945178536

GT(edge_aistpp/split_music/test_music):
Pitch Class Histogram Entropy sum (PHE): 2.844302882566178
'''

import librosa
import numpy as np
import os

def pitch_class_histogram_entropy(audio_file, hop_size=512):
    # 加载音频文件
    y, sr = librosa.load(audio_file)

    # 提取音高信息
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, hop_length=hop_size)

    # 将频率转换为音高类别
    pitch_classes = np.argmax(pitches, axis=0)

    # 计算音高直方图
    hist, _ = np.histogram(pitch_classes, bins=range(64))

    # 计算概率分布
    prob_dist = hist / np.sum(hist)

    # 计算熵
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))

    return entropy

# def grooving_pattern_similarity(audio_file, hop_size=512):
#     # 加载音频文件
#     y, sr = librosa.load(audio_file)
#
#     # 提取音频的节奏特征（例如：onset envelope）
#     onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_size)
#
#     # 对节奏特征进行归一化
#     onset_env /= onset_env.max()
#
#     # 计算 GS 分数（示例中简单地取节奏特征的均值）
#     gs_score = np.mean(onset_env)
#
#     return gs_score

# 示例用法
#########################################################################音频文件路径
# path = '../Generate/1221_origin_latent_unet_newmotclsloss_2/wav/'
# path = '../Generate/audio_result_ait_low_0124_merge/wav'
# path = '../Generate/audio_result_ait_low_0124_merge_2/wav'
# path = '../Generate/0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss/wav'
path = '../Generate/0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2/wav'
# path = '../edge_aistpp/split_music/test_music' # 运行需要注释掉下面的filename = filename.split("'")[1]


#########################################################################


fileList = os.listdir(path)
seq_names =[]
for filename in fileList:
    filename = filename.split("'")[1]

    seq_names.append(filename)


i=0
gs_result_sum=0
phe_result_sum=0
for seq_name in seq_names:

    # Produce a batch of log mel spectrogram examples.
    generate_audio_file = f'{path}/{fileList[i]}'
    # wav_file_r = f'wavs_clip_norepeat/{seq_name}.wav'  #
    # GT_audio_file = f'../edge_aistpp/wavs/{seq_name}.wav'
    i=i+1

    # gs_result = grooving_pattern_similarity(generate_audio_file)
    phe_result = pitch_class_histogram_entropy(generate_audio_file)

    print(f"Pitch Class Histogram Entropy (PHE): {phe_result}")




    phe_result_sum+=phe_result

print(f"Pitch Class Histogram Entropy sum (PHE): {phe_result_sum/len(seq_names)}")
