import librosa
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载音频文件
audio_path = "../edge_aistpp/wavs_clip_norepeat/gBR_sBM_cAll_d04_mBR0_ch01_segm_0.wav"
x, sr = librosa.load(audio_path, sr=None)

x_audio = x

# 2. 计算波形图
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(np.arange(len(x))/sr, x, color="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

# 3. 设置背景透明
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# 4. 保存为 PNG 格式
plt.savefig("waveform.png", transparent=True)


# 生成正态分布的高斯噪声
mu = 0  # 均值
sigma = 1  # 标准差
x = np.random.normal(mu, sigma, 262145)
x_noise = x
# 绘制波形图
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(np.arange(len(x))/sr,x, color="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

# 设置背景透明
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# 保存为 PNG 格式
plt.savefig("noise_waveform.png", transparent=True)

# # 加载音频文件
# audio_path = "your_audio.wav"
# x_audio, sr = librosa.load(audio_path, sr=None)
#
# # 生成正态分布的高斯噪声
# mu = 0  # 均值
# sigma = 1  # 标准差
# x_noise = np.random.normal(mu, sigma, len(x_audio))

# 混合音频和噪声
x = x_audio + x_noise*0.7

# 绘制波形图
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(np.arange(len(x))/sr,x, color="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

# 设置背景透明
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# 保存为 PNG 格式
plt.savefig("mixed_waveform.png", transparent=True)

# 混合音频和噪声
x = x_audio + x_noise*0.2

# 绘制波形图
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(np.arange(len(x))/sr,x, color="black")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

# 设置背景透明
fig.patch.set_alpha(0)
ax.set_facecolor('none')

# 保存为 PNG 格式
plt.savefig("mixed_waveform_02.png", transparent=True)