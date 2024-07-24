import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# ''' 111用的方法,用分类头做类别损失 ,生成效果不好,diffusion不用了'''
# class MOT(nn.Module):
#     def __init__(self, * , dim, depth,num_classes , heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
#         super().__init__()
#         # image_height, image_width = pair(image_size)
#         # patch_height, patch_width = pair(patch_size)

#         # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         # num_patches = (image_height // patch_height) * (image_width // patch_width)
#         num_patches = 356
#         # patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         # self.to_patch_embedding = nn.Sequential(
#         #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#         #     nn.LayerNorm(patch_dim),
#         #     nn.Linear(patch_dim, dim),
#         #     nn.LayerNorm(dim),
#         # )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()
        

#         model = [
#             nn.Dropout(dropout),
#             nn.Conv1d(357, 64, kernel_size=3, stride=1, padding=1),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim, 768),
#             nn.Dropout(dropout),
#             ]
#         self.model = nn.Sequential(*model)

#         self.mlp_head = nn.Linear(dim, num_classes)

#     def forward(self, x):
#         # x = self.to_patch_embedding(img) #img:torch.Size([1, 3, 256, 256])
#         b, n, _ = x.shape # b:1 n:64 _:1024    #b:1 n:356 _:219

#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  #cls_tokens:torch.Size([1, 1, 1024])# #b:1 n:356 _:768
#         x = torch.cat((cls_tokens, x), dim=1) #x:torch.Size([1, 65, 1024]) # #b:1 n:357 _:768
#         x += self.pos_embedding[:, :(n + 1)] #x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768
#         x = self.dropout(x)

#         x = self.transformer(x)#x:torch.Size([b, 357, 219])# #b:1 n:357 _:768

#         x_cls = x[:,0,:]  #x: torch.Size([1, 1024])
#         # x = self.to_latent(x)

#         clas = self.mlp_head(x_cls.squeeze(1))
#         x = self.model(x) 


#         return x , clas


# 适用于代码train_con_116_latent_unet_MOT_origendata_219prompt 将模型的输入聪哥过过64*768改成了357*219
class MOT(nn.Module):
    def __init__(self, *, dim, depth, num_classes, heads, mlp_dim, pool='mean', channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_patches = 356
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # model = [
        #     nn.Dropout(dropout),
        #     nn.Conv1d(357, 64, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim, 768),
        #     nn.Dropout(dropout),
        # ]
        # self.model = nn.Sequential(*model)

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x = self.to_patch_embedding(img) #img:torch.Size([1, 3, 256, 256])
        b, n, _ = x.shape  # b:1 n:64 _:1024    #b:1 n:356 _:219

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d',
                            b=b)  # cls_tokens:torch.Size([1, 1, 1024])# #b:1 n:356 _:768
        x = torch.cat((cls_tokens, x), dim=1)  # x:torch.Size([1, 65, 1024]) # #b:1 n:357 _:768
        x += self.pos_embedding[:, :(n + 1)]  # x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768
        x = self.dropout(x)

        x = self.transformer(x)  # x:torch.Size([b, 357, 219])# #b:1 n:357 _:768

        x_cls = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # x: torch.Size([1, 1024])
        # x = self.to_latent(x)

        clas = self.mlp_head(x_cls)
        # x = self.model(x)

        return x, clas

''''

class MOT(nn.Module):
    def __init__(self, *, dim, depth, num_classes, heads, mlp_dim, pool='mean', channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_patches = 356
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        model = [
            nn.Dropout(dropout),
            nn.Conv1d(357, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 768),
            nn.Dropout(dropout),
        ]
        self.model = nn.Sequential(*model)

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x = self.to_patch_embedding(img) #img:torch.Size([1, 3, 256, 256])
        b, n, _ = x.shape  # b:1 n:64 _:1024    #b:1 n:356 _:219

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d',
                            b=b)  # cls_tokens:torch.Size([1, 1, 1024])# #b:1 n:356 _:768
        x = torch.cat((cls_tokens, x), dim=1)  # x:torch.Size([1, 65, 1024]) # #b:1 n:357 _:768
        x += self.pos_embedding[:, :(n + 1)]  # x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768
        x = self.dropout(x)

        x = self.transformer(x)  # x:torch.Size([b, 357, 219])# #b:1 n:357 _:768

        x_cls = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # x: torch.Size([1, 1024])
        # x = self.to_latent(x)

        clas = self.mlp_head(x_cls)
        x = self.model(x)

        return x, clas
    '''

'''
class MOT(nn.Module):
    def __init__(self, *, dim, depth, num_classes, heads, mlp_dim, pool='mean', channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_patches = 356
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        self.temp_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        model = [
            nn.Dropout(dropout),
            nn.Conv1d(357, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 768),
            nn.Dropout(dropout),
        ]
        self.model = nn.Sequential(*model)

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x = self.to_patch_embedding(img) #img:torch.Size([1, 3, 256, 256])
        b, n, _ = x.shape  # b:1 n:64 _:1024    #b:1 n:356 _:219

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d',
                            b=b)  # cls_tokens:torch.Size([1, 1, 1024])# #b:1 n:356 _:768
        x = torch.cat((cls_tokens, x), dim=1)  # x:torch.Size([1, 65, 1024]) # #b:1 n:357 _:768
        x += self.temp_embedding[:, :(n + 1)]  # x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768
        x = self.dropout(x)

        x = self.transformer(x)  # x:torch.Size([b, 357, 219])# #b:1 n:357 _:768

        x_cls = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # x: torch.Size([1, 1024])
        # x = self.to_latent(x)

        clas = self.mlp_head(x_cls)
        x = self.model(x)

        return x, clas
'''

''' 110用的方法,不是用的分类头做类别损失而是所有的一起用来做类别损失了,112继续用,因为111生成效果不好 
pool = 'mean' or pool = 'cls' .pool type must be either cls (cls token) or mean (mean pooling)




class MOT(nn.Module):
    def __init__(self, * , dim, depth,num_classes , heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_patches = 356
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        self.temp_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        

        model = [
            nn.Dropout(dropout),
            nn.Conv1d(357, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 768),
            nn.Dropout(dropout),
            ]
        self.model = nn.Sequential(*model)

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x = self.to_patch_embedding(img) #img:torch.Size([1, 3, 256, 256])
        b, n, _ = x.shape # b:1 n:64 _:1024    #b:1 n:356 _:219

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  #cls_tokens:torch.Size([1, 1, 1024])# #b:1 n:356 _:768
        x = torch.cat((cls_tokens, x), dim=1) #x:torch.Size([1, 65, 1024]) # #b:1 n:357 _:768
        x += self.temp_embedding[:, :(n + 1)] #x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768
        x = self.dropout(x)

        x = self.transformer(x)#x:torch.Size([b, 357, 219])# #b:1 n:357 _:768

        x_cls = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  #x: torch.Size([1, 1024])
        # x = self.to_latent(x)

        clas = self.mlp_head(x_cls)
        x = self.model(x) 


        return x , clas
'''
        

    # def forward(self, x):####################这是错误的!!!!!!!!!! 分类层不能只是训练一个没有用到的线性层!!
    #     # x = self.to_patch_embedding(img) #img:torch.Size([1, 3, 256, 256])
    #     b, n, _ = x.shape # b:1 n:64 _:1024    #b:1 n:356 _:219

    #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  #cls_tokens:torch.Size([1, 1, 1024])# #b:1 n:356 _:768
    #     x = torch.cat((cls_tokens, x), dim=1) #x:torch.Size([1, 65, 1024]) # #b:1 n:357 _:768
    #     x += self.pos_embedding[:, :(n + 1)] #x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768
    #     x = self.dropout(x)

    #     x = self.transformer(x)#x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768

    #     # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  #x: torch.Size([1, 1024])
    #     # x = self.to_latent(x)

    #     clas = self.mlp_head(cls_tokens)
    #     x = self.model(x)

    #     return x , clas

    # def forward(self, x):
    #     # x = self.to_patch_embedding(img) #img:torch.Size([1, 3, 256, 256])
    #     b, n, _ = x.shape # b:1 n:64 _:1024    #b:1 n:356 _:219

    #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  #cls_tokens:torch.Size([1, 1, 1024])# #b:1 n:356 _:768
    #     x = torch.cat((cls_tokens, x), dim=1) #x:torch.Size([1, 65, 1024]) # #b:1 n:357 _:768
    #     x += self.pos_embedding[:, :(n + 1)] #x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768
    #     x = self.dropout(x)

    #     x = self.transformer(x)#x:torch.Size([1, 65, 1024])# #b:1 n:357 _:768

    #     x_cls = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  #x: torch.Size([1, 1024])
    #     # x = self.to_latent(x)

    #     clas = self.mlp_head(x_cls)
    #     x = self.model(x)


    #     return x , clas