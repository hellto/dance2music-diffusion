'''
0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2 0.190
这是使用mean的mot_classhead方法,不使用genre loss加以训练的方法

1224_origin_latent_unet_resnet_newmotclsloss_2 0.265
将mot换成resnet的方法 使用latent的方法

0108_mean_(1221cls)_origin_latent_unet_newmotclsloss 0.216
使用mean的mot_classhead方法,训练的mot,和latent的方法

0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss 0.214
使用mean的mot_classhead方法,训练的mot,和latent的方法

1221_origin_latent_unet_newmotclsloss_2/wav/ 0.217
第一次使用上latent classhead只使用第一个

Generate/audio_result_ait_low_0124_merge/wav 0.214
D2M-GAN的生成结果

1204_mot_class_origen_unet_final_origen_dataset/wav 0.196
不使用latent的方法DMAE 使用mot

服务器训练的结果/0606epoch450 0.252
使用resnet和 不使用latent

服务器训练的结果/0606epoch400 0.252
使用resnet和 不使用latent

服务器训练的结果/1218_motd12_only_epoch_1200 0.212
不使用latent的方法DMAE 使用mot

cdcd_RESULT 0.194

0129_mean_(1221cls)_origin_latent_unet_newmot_mean_clsloss 0.225

gt 0.285

SGP新家谱会议论文的可视化

'''

from absl import app
from absl import flags
from absl import logging

import matplotlib.pyplot as plt

import os
from librosa import beat
import librosa
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal
from aist_plusplus.loader import AISTDataset


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir', '../edge_aistpp/',
    'Path to the AIST++ annotation files.')
flags.DEFINE_string(
    # 'audio_dir', '../服务器训练的结果/0606epoch450/wav',
    # 'audio_dir', '../服务器训练的结果/0606epoch450/wav',
# 'audio_dir', '../服务器训练的结果/1218_motd12_only_epoch_1200/wav',
#     'audio_dir', '../服务器训练的结果/1204_mot_class_origen_unet_final_origen_dataset/wav/',
# 'audio_dir', '../Generate/audio_result_ait_low_0124_merge/wav',
# 'audio_dir', '../Generate/1221_origin_latent_unet_newmotclsloss_2/wav/',
# 'audio_dir', '../Generate/0108_mean_(1221cls)_origin_latent!_unet_newmotclsloss/wav',
# 'audio_dir', '../Generate/1224_origin_latent_unet_resnet_newmotclsloss_2/wav',
# 'audio_dir', '../Generate/0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2/wav',
# 'audio_dir', '../Generate/cdcd_RESULT/wav',
# 'audio_dir', '../Generate/0129_mean_(1221cls)_origin_latent_unet_newmot_mean_clsloss/wav',
# 'audio_dir', '../edge_aistpp/split_music/test_music',
'audio_dir', '../Generate/dmd_sgp_test/wav',#dmd_sgp
    'Path to the AIST wav files.')
flags.DEFINE_string(
    # 'audio_cache_dir', '../服务器训练的结果/0606epoch450/feature/',
    # 'audio_cache_dir', '../服务器训练的结果/0606epoch400/feature/',
    # 'audio_cache_dir', '../Generate/train_con_66tensorboard/test4_aist_audio_feats/',
# 'audio_cache_dir', '../服务器训练的结果/1204_mot_class_origen_unet_final_origen_dataset/aist_audio_feats/',
# 'audio_cache_dir', '../Generate/audio_result_ait_low_0124_merge/feature/',
# 'audio_cache_dir', '../Generate/1221_origin_latent_unet_newmotclsloss_2/feature/',
# 'audio_cache_dir', '../Generate/0108_mean_(1221cls)_origin_latent_unet_newmotclsloss/feature/',
# 'audio_cache_dir', '../Generate/cdcd_RESULT/feature/',
# 'audio_cache_dir', '../Generate/1224_origin_latent_unet_resnet_newmotclsloss_2/feature/',
# 'audio_cache_dir', '../Generate/0124_mean_(1221cls)_origin_latent!_unet_nogenreloss_newmotclsloss_2/feature/',
# 'audio_cache_dir', '../Generate/0129_mean_(1221cls)_origin_latent_unet_newmot_mean_clsloss/feature/',
'audio_cache_dir', '../Generate/dmd_sgp_test/feature/',

    'Path to cache dictionary for audio features.')
flags.DEFINE_enum(
    'split', 'testval', ['train', 'testval'],
    'Whether do training set or testval set.')
flags.DEFINE_string(
    'result_files', '../mnt/data/aist_paper_results/*.pkl',
    'The path pattern of the result files.')
flags.DEFINE_bool(
    'legacy', True,
    'Whether the result files are the legacy version.')


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
    return keypoints3d


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32) #  joints (256,24,3)   [[[-0.0029631   1.7059059  -0.2154109 ],
    velocity[1:] = joints[1:] - joints[:-1]  # velocity (356,24,3)
    velocity_norms = np.linalg.norm(velocity, axis=2) #velocity_norms(356,24)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,) #envelope(356,)

#####################################################
    plt.plot(envelope) #可视化运动速度的包络
    # plt.show()
######################################################

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0

#########################################
    for i in range(356) :
        if peak_onehot[i] == 1 :
            plt.axvline(i, linewidth=2, alpha=0.6,linestyle="dashed", color="green")
    # plt.axvline(peak_idxs, linestyle="dashed", color="red")
######################################

    return peak_onehot


def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)



RNG = np.random.RandomState(42)



def close_tfrecord_writers(writers):
    for w in writers:
        w.close()


def write_tfexample(writers, tf_example):
    random_writer_idx = RNG.randint(0, len(writers))
    writers[random_writer_idx].write(tf_example.SerializeToString())


def load_cached_audio_features(seq_name):
    audio_name = seq_name.split("_")[-2]
    return np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy")), audio_name


def cache_audio_features(seq_names_fulls):
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    def _get_tempo(audio_name):
        """Get tempo (BPM) for a music by parsing music name."""
        assert len(audio_name) == 4
        if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return int(audio_name[3]) * 10 + 80
        elif audio_name[0:3] == 'mHO':
            return int(audio_name[3]) * 5 + 110
        else:
            assert False, audio_name

    # # seq_names=seq_names_fulls
    # seq_names=list(set([seq_names_full.split("'")[1] for seq_names_full in seq_names_fulls]))
    # audio_names = list(set([seq_name.split("_")[-2] for seq_name in seq_names]))
    # # audio_names = list(set([seq_name for seq_name in seq_names]))
    #
    # for seq_names_full in seq_names_fulls:
    #     seq_name=seq_names_full.split("'")[1]

    # seq_names=seq_names_fulls
    seq_names = list(set([seq_names_full.split(".")[0] for seq_names_full in seq_names_fulls]))
    audio_names = list(set([seq_name.split("_")[-2] for seq_name in seq_names]))
    # audio_names = list(set([seq_name for seq_name in seq_names]))

    for seq_names_full in seq_names_fulls:
        # seq_name=seq_names_full.split("'")[1]
        seq_name = seq_names_full.split(".")[0]  # dmd_sgp独有
        save_path = os.path.join(FLAGS.audio_cache_dir, f"{seq_name}.npy")
        if os.path.exists(save_path):
            continue
        data, _ = librosa.load(os.path.join(FLAGS.audio_dir, f"{seq_names_full}"), sr=SR)

        # data = librosa.decompose.hpss(data)[0]###音频增益
        # data = librosa.effects.preemphasis(data)###音频去噪
        # data = librosa.util.normalize(data)###音频均衡化

        envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
        # envelope = librosa.onset.onset_strength()  # (seq_len,)
        mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
        chroma = librosa.feature.chroma_cens(
            y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)
        audio_name = seq_name.split("_")[-2]
        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
            start_bpm=_get_tempo(audio_name), tightness=100)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)


        audio_feature = np.concatenate([
            envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)
        np.save(save_path, audio_feature)


# def main(_):
#
#
#     # 设定文件路径
#     path = FLAGS.audio_dir
#     # 获取该目录下所有文件，存入列表中
#     seq_names_full = os.listdir(path)
#     # create audio features
#     print("Pre-compute audio features ...")
#     os.makedirs(FLAGS.audio_cache_dir, exist_ok=True)
#     cache_audio_features(seq_names_full)
#


def main(_):
    import glob
    import tqdm
    from smplx import SMPL

    # set smpl
    smpl = SMPL(model_path="../evaluate/smpl/", gender='MALE', batch_size=1)
    # smpl = SMPL(model_path="/mnt/data/aist_plusplus_final/motions/", gender='MALE', batch_size=1)

    # 设定文件路径
    path = FLAGS.audio_dir
    # 获取该目录下所有文件，存入列表中
    seq_names_full = os.listdir(path)
    # create audio features
    print("Pre-compute audio features ...")
    os.makedirs(FLAGS.audio_cache_dir, exist_ok=True)
    cache_audio_features(seq_names_full)
    print("Pre-compute audio features finish!")
#####################################################################


    # create list
    seq_names = []
    # 设定文件路径
    path = FLAGS.audio_dir
    # 获取该目录下所有文件，存入列表中
    fileList = os.listdir(path)

    #
    # for filename in fileList:
    #     name = filename.split("'")[1]
    #     seq_names.append(name)
    for filename in fileList:
        # name = filename.split("'")[1]
        name = filename.split(".")[0] #sgp_dmd独有
        seq_names.append(name)


    # calculate score on real data
    dataset = AISTDataset(FLAGS.anno_dir)
    n_samples = len(seq_names)
    beat_scores = []
    # for i, seq_name in enumerate(seq_names):

    seq_name="gLO_sBM_cAll_d13_mLO2_ch02"
    # logging.info("processing %d / %d" % (i + 1, n_samples))
    # get real data motion beats
    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
        dataset.motion_dir, seq_name)
    smpl_trans /= smpl_scaling
    keypoints3d = smpl.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:356, :24, :]   # (seq_len, 24, 3)
    motion_beats = motion_peak_onehot(keypoints3d)
    # get real data music beats
    # audio_name = seq_name.split("_")[4]############这里有一个很大的问题我不需要文件名变成音乐名称
    # audio_feature = np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy"))
    # audio_feature = np.load(os.path.join(FLAGS.audio_cache_dir, f"{seq_name}.npy"))#########f"{audio_name}.npy"
    # audio_beats = audio_feature[:keypoints3d.shape[0], -1] # last dim is the music beats
    #
    # data, _ = librosa.load(os.path.join(FLAGS.audio_dir, f"{seq_names_full}"), sr=SR)
    # envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    # beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    # beat_onehot[beat_idxs] = 1.0  # (seq_len,)

#######################
    # audio_path='../参考文献可视化结果/cmt/1.demo-new(Av848550858,P1).mp3'##########
    # audio_path = '../参考文献可视化结果/cmt/1.animation(Av208625251,P1).mp3'  ##########
    audio_path = '../参考文献可视化结果/cmt/1.wzk_vlog_beat_enhance1_track1238(Av891206601,P1).mp3'  ##########
    # audio_path = '../参考文献可视化结果/dance2music/slow_1.mp3'  ##########
    # audio_path = '../参考文献可视化结果/dance2midi/video00045_0_4160_2160.mp3'



    y, sr = librosa.load(audio_path)  # y:ndarray  sr=22050
    y = y[:131072]
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # plt.axvline(librosa.frames_to_time(beat_frames), np.repeat(0, len(beat_frames)),linewidth=2, alpha=0.6,linestyle="dashed", color="red")

    # plt.axvline(librosa.frames_to_time(beat_frames),  linewidth=2, alpha=0.6,linestyle="dashed", color="red")

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Plotting
    # plt.figure(figsize=(10, 4))
    # librosa.display.waveshow(y, sr=sr, alpha=0.6)
    for beat_time in beat_times:
        plt.axvline(beat_time*60, 0, 1, linewidth=2, alpha=0.6, linestyle="dashed", color="red")

    # audio_beats = librosa.frames_to_time(beat_frames)*60
    # for i in range(356):
    #     if audio_beats[i]==1 :
    #         plt.axvline(i, linewidth=2, alpha=0.6,linestyle="dashed", color="red")
    # plt.show()
    plt.savefig(f'SGP_DMD_{seq_name}.png')
    plt.clf()
    beat_score = alignment_score(beat_times*60, motion_beats, sigma=3)
    print("\nBeat score on real data: %.3f\n" , beat_score)
#############################
        # get beat alignment scores
    # beat_score = alignment_score(audio_beats, motion_beats, sigma=3)
    #     # beat_scores.append(beat_score)
    # print ("\nBeat score on real data: %.3f\n" % beat_scores )
    # calculate score on generated music data
    # result_files = sorted(glob.glob(FLAGS.result_files))
    # result_files = [f for f in result_files if f[-8:-4] in f[:-8]]
    # if FLAGS.legacy:
    #     # for some reason there are repetitive results. Skip them
    #     result_files = {f[-34:]: f for f in result_files}
    #     result_files = result_files.values()
    # n_samples = len(result_files)
    # beat_scores = []
    # for result_file in tqdm.tqdm(result_files):
    #     if FLAGS.legacy:
    #         with open(result_file, "rb") as f:
    #             data = pickle.load(f)
    #         result_motion = np.concatenate([
    #             np.pad(data["pred_trans"], ((0, 0), (0, 0), (6, 0))),
    #             data["pred_motion"].reshape(1, -1, 24 * 9)
    #         ], axis=-1)  # [1, 120 + 1200, 225]
    #     else:
    #         result_motion = np.load(result_file)[None, ...]  # [1, 120 + 1200, 225]
    #     keypoints3d = recover_motion_to_keypoints(result_motion, smpl)
    #     motion_beats = motion_peak_onehot(keypoints3d)
    #     if FLAGS.legacy:
    #         audio_beats = data["audio_beats"][0] > 0.5
    # #     else:
    # #         audio_name = result_file[-8:-4]
    # #         audio_feature = np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy"))
    # #         audio_beats = audio_feature[:, -1] # last dim is the music beats
    # #     beat_score = alignment_score(audio_beats[120:], motion_beats[120:], sigma=3)
    # #     beat_scores.append(beat_score)
    # # print ("\nBeat score on generated data: %.3f\n" % (sum(beat_scores) / n_samples))
    #

if __name__ == '__main__':
    app.run(main)
