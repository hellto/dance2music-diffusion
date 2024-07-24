# import matplotlib.pyplot as plt
# import numpy as np
#
# def draw_transformer_encoder(num_layers):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 10)
#     ax.set_aspect('equal')
#
#     # Draw encoder layers
#     for i in range(num_layers):
#         layer_y = 9 - i * 2
#         ax.text(1, layer_y, f'Encoder Layer {i+1}', fontsize=12, ha='left', va='center', color='black')
#         ax.add_patch(plt.Rectangle((2, layer_y - 0.5), 6, 1, edgecolor='black', facecolor='lightgrey'))
#         ax.text(3, layer_y, 'Multi-Head\nSelf-Attention', fontsize=10, ha='left', va='center', color='black')
#         ax.text(3, layer_y - 0.5, 'Layer Normalization', fontsize=10, ha='left', va='top', color='black')
#         ax.text(3, layer_y + 0.5, 'Layer Normalization', fontsize=10, ha='left', va='bottom', color='black')
#         ax.text(7, layer_y, 'Position-wise\nFeed-Forward\nNetwork', fontsize=10, ha='left', va='center', color='black')
#         ax.text(7, layer_y - 0.5, 'Layer Normalization', fontsize=10, ha='left', va='top', color='black')
#         ax.text(7, layer_y + 0.5, 'Layer Normalization', fontsize=10, ha='left', va='bottom', color='black')
#         if i < num_layers - 1:
#             ax.arrow(8.5, layer_y, 0.5, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
#
#     plt.title('Transformer Encoder', fontsize=14)
#     plt.axis('off')
#     plt.show()
#
# # Draw a Transformer encoder with 6 layers
# draw_transformer_encoder(1)

import matplotlib.pyplot as plt

# 创建一个图形和一个子图
fig, ax = plt.subplots()

# 设置标题
ax.set_title('Transformer Architecture')

# 绘制输入端
ax.text(0.5, 0.9, 'Input', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
ax.text(0.5, 0.7, 'Embedding', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))

# 绘制Encoder
for i in range(4):
    ax.text(0.1, 0.5 - i*0.1, f'Encoder Layer {i+1}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
    ax.plot([0.3, 0.7], [0.5 - i*0.1, 0.5 - i*0.1], color='gray', linestyle='-', linewidth=2)

# 绘制Decoder
for i in range(4):
    ax.text(0.9, 0.5 - i*0.1, f'Decoder Layer {i+1}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightcoral', alpha=0.5))
    ax.plot([0.7, 1.1], [0.5 - i*0.1, 0.5 - i*0.1], color='gray', linestyle='-', linewidth=2)

# 绘制输出端
ax.text(0.5, 0.1, 'Output', ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))

# 移除坐标轴
ax.axis('off')

# 显示图形
plt.show()