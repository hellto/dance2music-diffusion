'''
计算beat覆盖率

服务器训练的结果/1219_mot_genretoken_only_epoch_400/wav
Beat Coverage Scores sum (BCS): 1.1585213032581454
Beat hit Scores sum (BCS): 0.42552213868003347
bAVER: 0.7986495126524235

1221_origin_latent_unet_newmotclsloss_2/wav/:
Number of aligned beats sum: 82
Beat Coverage Scores sum (BCS): 1.2813074352548035
Beat hit Scores sum (BCS): 0.46911027568922303
bAVER: 0.8969855629767067

audio_result_ait_low_0124_merge:
Number of aligned beats sum: 103
Beat Coverage Scores sum (BCS): 1.1897817460317461
Beat hit Scores sum (BCS): 0.5679761904761904
bAVER: 0.8884809467046133

audio_result_ait_low_0124_merge_2:
Number of aligned beats sum: 98
Beat Coverage Scores sum (BCS): 1.1822263993316626
Beat hit Scores sum (BCS): 0.5686299081035923
bAVER: 0.8842578622908175

0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss:
Beat Coverage Scores sum (BCS): 1.203988095238095
Beat hit Scores sum (BCS): 0.46690476190476193
bAVER: 0.8465941583301642

0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2
Beat Coverage Scores sum (BCS): 1.2579365079365081
Beat hit Scores sum (BCS): 0.5881150793650795
bAVER: 0.9411858601778533

GT:
Beat Coverage Scores sum (BCS): 1.0
Beat hit Scores sum (BCS): 0.9732142857142858
bAVER: 4.181135192322468

0129_mean_(1221cls)_origin_latent_unet_newmot_mean_clsloss:
Beat Coverage Scores sum (BCS): 1.1621230158730158
Beat hit Scores sum (BCS): 0.4270436507936508
bAVER: 0.8015242752431183

cdcd_RESULT:
Beat Coverage Scores sum (BCS): 0.932420634920635
Beat hit Scores sum (BCS): 0.43783730158730166
bAVER: 3.67202236649638

1224_origin_latent_unet_resnet_newmotclsloss_2: resnet+DMAE
Number of aligned beats sum: 86
Beat Coverage Scores sum (BCS): 1.189076858813701
Beat hit Scores sum (BCS): 0.49974937343358394
bAVER: 0.8539415890396018

1219_mot_genretoken_only_epoch_5000:  MOT+NO_DMAE
Beat Coverage Scores sum (BCS): 1.1585213032581454
Beat hit Scores sum (BCS): 0.42552213868003347
bAVER: 0.7986495126524235

0220_mean_(1221cls)_origin_latent_unet_newmotclsloss_357219prompt/wav:没有使用融合模块
Beat Coverage Scores sum (BCS): 1.1833134920634918
Beat hit Scores sum (BCS): 0.4325396825396825
bAVER: 0.8168652977098615

'''

import librosa
import os
import numpy as np

# path = '../Generate/resnet_diffusion消融实验/10'
# Beat Coverage Scores sum (BCS): 1.1878028404344192
# Beat hit Scores sum (BCS): 0.5365497076023392
# bAVER: 0.8715726538173705

# path = '../Generate/resnet_diffusion消融实验/40'
# Beat Coverage Scores sum (BCS): 1.24312865497076
# Beat hit Scores sum (BCS): 0.5667293233082705
# bAVER: 0.9209810009481343

# path = '../Generate/resnet_diffusion消融实验/100'
# Beat Coverage Scores sum (BCS): 1.1804093567251461
# Beat hit Scores sum (BCS): 0.5117585630743525
# bAVER: 0.8547330577497585

# path = '../Generate/resnet_diffusion消融实验/200'
# Beat Coverage Scores sum (BCS): 1.2080409356725148
# Beat hit Scores sum (BCS): 0.5445071010860485
# bAVER: 0.8878853361028124

# path = '../Generate/resnet_diffusion消融实验/400'
# Beat Coverage Scores sum (BCS): 1.2080409356725148
# Beat hit Scores sum (BCS): 0.5369883040935672
# bAVER: 0.8841259376065718


# path = '../Generate/resnet_diffusion消融实验/500'
# Beat Coverage Scores sum (BCS): 1.2080409356725148
# Beat hit Scores sum (BCS): 0.5369883040935672
# bAVER: 0.8841259376065718




# path = '../Generate/resnet_diffusion消融实验/10'
# path = '../Generate/resnet_diffusion消融实验/40'
# path = '../Generate/resnet_diffusion消融实验/100'
# path = '../Generate/resnet_diffusion消融实验/200'
# path = '../Generate/resnet_diffusion消融实验/400'
# path = '../Generate/resnet_diffusion消融实验/500'

#########################################################################
# path = '../Generate/1221_origin_latent_unet_newmotclsloss_2/wav/'
# path = '../Generate/audio_result_ait_low_0124_merge/wav'
# path = '../Generate/audio_result_ait_low_0124_merge_2/wav'

# path = '../Generate/0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss/wav'
# path = '../Generate/0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2/wav'
# path = '../edge_aistpp/split_music/test_music' # 运行需要注释掉下面的filename = filename.split("'")[1]
# path ='../Generate/0129_mean_(1221cls)_origin_latent_unet_newmot_mean_clsloss'
# path ='../Generate/cdcd_RESULT'

# path = '../Generate/1224_origin_latent_unet_resnet_newmotclsloss_2/wav'
# path ='../服务器训练的结果/1219_mot_genretoken_only_epoch_400/wav'

# path = '../Generate/0220_mean_(1221cls)_origin_latent_unet_newmotclsloss_357219prompt/wav'


#########################################################################





def detect_beats(audio_file):
    # 加载音频文件
    y, sr = librosa.load(audio_file)

    # 使用节拍追踪算法检测节拍
    tempo, beats = librosa.beat.beat_track(y=y[0:131073], sr=sr)

    return tempo, beats



def calculate_bcs(bg, bt, ba):
    """
    Calculate Beat Coverage Scores (BCS) based on the given parameters.

    Parameters:
    - bg: Number of detected beats in the generated music.
    - bt: Total number of beats in the ground truth.
    - ba: Number of aligned beats in the generated music.

    Returns:
    - BCS: Beat Coverage Scores.
    """
    try:
        bcs = bg / bt
        return bcs
    except ZeroDivisionError:
        # Handle the case where total beats in ground truth (bt) is zero.
        print("Error: Total beats in ground truth (bt) should not be zero.")
        return None

def calculate_bhs(bg, bt, ba):
    """
    Calculate Beat Coverage Scores (BCS) based on the given parameters.

    Parameters:
    - bg: Number of detected beats in the generated music.
    - bt: Total number of beats in the ground truth.
    - ba: Number of aligned beats in the generated music.

    Returns:
    - BCS: Beat Coverage Scores.
    """
    try:
        bhs = ba / bt
        return bhs
    except ZeroDivisionError:
        # Handle the case where total beats in ground truth (bt) is zero.
        print("Error: Total beats in beat align (ba) should not be zero.")
        return None


def count_aligned_beats(audio_file1, audio_file2, tolerance=5):
    """
    计算两段音频中的节拍对齐数量。

    Parameters:
    - audio_file1: 第一个音频文件路径。
    - audio_file2: 第二个音频文件路径。
    - tolerance: 容忍的时间差，单位为秒。

    Returns:
    - aligned_count: 节拍对齐数量。
    """
    # 加载音频文件并检测节拍
    y1, sr1 = librosa.load(audio_file1)
    y2, sr2 = librosa.load(audio_file2)

    _, beats1 = librosa.beat.beat_track(y=y1, sr=sr1)
    _, beats2 = librosa.beat.beat_track(y=y2[0:131073], sr=sr2)

    # 计算对齐数量
    aligned_count = 0
    for beat1 in beats1:
        for beat2 in beats2:
            if abs(beat1 - beat2) <= tolerance:
                aligned_count += 1

    return aligned_count




fileList = os.listdir(path)
seq_names =[]
for filename in fileList:
    filename = filename.split("'")[1]  ##########如果是测试test_music 要去掉这一行

    seq_names.append(filename)


i=0
bcs_result_sum =0
bhs_result_sum =0
aligned_count_sum = 0
for seq_name in seq_names:

    # Produce a batch of log mel spectrogram examples.
    generate_audio_file = f'{path}/{fileList[i]}'
    # wav_file_r = f'wavs_clip_norepeat/{seq_name}.wav'  #
    GT_audio_file = f'../edge_aistpp/wavs/{seq_name}.wav'
    # GT_audio_file = f'../edge_aistpp/wavs/{seq_name}' #########如果是测试test_music 用这一行
    i=i+1

    generate_tempo, generate_beats = detect_beats(generate_audio_file)
    GT_tempo, GT_beats = detect_beats(GT_audio_file)


    # print(f"Detected generate Tempo: {generate_tempo} BPM")
    # print(f"Number of generate Beats Detected: {len(generate_beats)}")

    tolerance = 5  # 容忍的时间差，单位为秒
    aligned_count = count_aligned_beats(generate_audio_file, GT_audio_file, tolerance)

    # 示例用法：
    bg = len(generate_beats)  # 生成音乐中检测到的拍数
    bt = len(GT_beats)  # 地面实况中的总拍数
    ba = aligned_count  # aligned_count 生成音乐中与地面实况对齐的拍数



    bcs_result = calculate_bcs(bg, bt, ba)
    bhs_result = calculate_bhs(bg, bt, ba)
    if bcs_result is not None:
        print(f"Number of generate_beats: {bg}")
        print(f"Number of GT_beats: {bt}")
        print(f"Number of aligned beats: {aligned_count}")
        print(f"Beat Coverage Scores (BCS): {bcs_result}")
        print(f"Beat hit Scores (BCS): {bhs_result}")
        aligned_count_sum+=aligned_count
        bcs_result_sum+=bcs_result
        bhs_result_sum+=bhs_result


print(f"Number of aligned beats sum: {aligned_count_sum}")
print(f"Beat Coverage Scores sum (BCS): {bcs_result_sum/len(seq_names)}")
print(f"Beat hit Scores sum (BCS): {bhs_result_sum/len(seq_names)}")
if (bcs_result_sum/len(seq_names)) <= 1:
    bAVER=0.5*(np.exp(bcs_result_sum/len(seq_names)+1)+bhs_result_sum/len(seq_names))
if (bcs_result_sum/len(seq_names)) > 1:
    bAVER=0.5*(np.exp(bcs_result_sum/len(seq_names)-1)+bhs_result_sum/len(seq_names))
print(f"bAVER: {bAVER}")