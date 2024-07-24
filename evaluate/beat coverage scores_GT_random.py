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
GT random:
Beat Coverage Scores sum (BCS): 1.068015873015873
Beat hit Scores sum (BCS): 0.5567460317460318
bAVER: 0.8135641651425896
Beat Coverage Scores sum (BCS): 1.096706349206349
Beat hit Scores sum (BCS): 0.7229563492063492
bAVER: 0.9122466040783239
Beat Coverage Scores sum (BCS): 0.9799404761904762
Beat hit Scores sum (BCS): 0.44855158730158734
bAVER: 3.8454317348196447


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


cmt_generate_wav_npz_0: cmt midi合成transformer
Beat Coverage Scores sum (BCS): 0.8664995822890559
Beat hit Scores sum (BCS): 0.37619047619047613
bAVER: 3.420907415904383
cmt_generate_wav_npz_1: cmt midi合成transformer
Beat Coverage Scores sum (BCS): 0.8054093567251462
Beat hit Scores sum (BCS): 0.41610275689223053
bAVER: 3.2492817960512737
cmt_generate_wav_npz_2: cmt midi合成transformer
Beat Coverage Scores sum (BCS): 0.914264828738513
Beat hit Scores sum (BCS): 0.4326023391812866
bAVER: 3.607276703732813
Beat Coverage Scores sum (BCS): 0.9725146198830409
Beat hit Scores sum (BCS): 0.43546365914786966
bAVER: 3.812097181699269
Beat Coverage Scores sum (BCS): 0.8727234753550543
Beat hit Scores sum (BCS): 0.33546365914786963
bAVER: 3.42072742925648
'''

import librosa
import os
import numpy as np
import random

#########################################################################
# path = '../Generate/1221_origin_latent_unet_newmotclsloss_2/wav/'
# path = '../Generate/audio_result_ait_low_0124_merge/wav'
# path = '../Generate/audio_result_ait_low_0124_merge_2/wav'

# path = '../Generate/0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss/wav'
# path = '../Generate/0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2/wav'
path = '../edge_aistpp/split_music/test_music' # 运行需要注释掉下面的filename = filename.split("'")[1]
# path ='../Generate/0129_mean_(1221cls)_origin_latent_unet_newmot_mean_clsloss'
# path ='../Generate/cdcd_RESULT'

# path = '../Generate/1224_origin_latent_unet_resnet_newmotclsloss_2/wav'
# path ='../服务器训练的结果/1219_mot_genretoken_only_epoch_400/wav'

# path = '../Generate/0220_mean_(1221cls)_origin_latent_unet_newmotclsloss_357219prompt/wav'


# path ='../Generate/cmt_generate_wav/cmt_generate_wav_npz_4'

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
    # filename = filename.split("'")[1]  ##########如果是测试test_music 要去掉这一行
    # filename = filename.split('.')[0].split("'")[2]  ##########如果是测试test_music 要去掉这一行
    # filename = filename.split('_')[0]+ "_"+ filename.split('_')[1]+ "_cAll_"+filename.split('_')[3]+"_"+filename.split('_')[4]+"_"+filename.split('_')[5]

    filename = filename.split(".")[0]  ##########如果是测试test_music 要去掉这一行
    seq_names.append(filename)


i=0
bcs_result_sum =0
bhs_result_sum =0
aligned_count_sum = 0
for seq_name in seq_names:

    # Produce a batch of log mel spectrogram examples.

    random_inter = random.randint(0,19)
    generate_audio_file = f'{path}/{fileList[random_inter]}'
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