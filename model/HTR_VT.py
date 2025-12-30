import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath
import torchvision.models as models

import numpy as np
from model import resnet18 # Your original custom resnet wrapper
from functools import partial


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches
        self.bias = torch.ones(1, 1, self.num_patches, self.num_patches)
        self.back_bias = torch.triu(self.bias)
        self.forward_bias = torch.tril(self.bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_weights = self.attn_drop(attn) # <--- Keep this

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights # <--- Return the weights as a second output

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            num_patches,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.0,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        self.attn = Attention(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_attention=False):
        # 1. Run Attention (returns feature + weights)
        x_attn, attn_weights = self.attn(self.norm1(x))
        
        # 2. Add Residual (Use x_attn, NOT the tuple)
        x = x + self.drop_path1(self.ls1(x_attn))

        # 3. MLP Block
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        if return_attention:
            return attn_weights
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

# --- HELPER: ADAPTER FOR TORCHVISION MODELS ---
class TorchVisionAdapter(nn.Module):
    """
    Adapts ResNet/VGG features to match the expected (Batch, Channels, Height, Width)
    and then projects them to (Batch, Embed_Dim, H, W) so they fit the ViT.
    """
    def __init__(self, backbone_name, embed_dim):
        super().__init__()
        
        if backbone_name == 'resnet18':
            m = models.resnet18(pretrained=True)
            # Remove AvgPool and FC, keep spatial features
            self.features = nn.Sequential(*list(m.children())[:-2]) 
            self.out_channels = 512
            
        elif backbone_name == 'resnet50':
            m = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(m.children())[:-2])
            self.out_channels = 2048
            
        elif backbone_name == 'vgg16':
            m = models.vgg16(pretrained=True)
            self.features = m.features
            self.out_channels = 512
        
        # Projection layer to match ViT embed_dim
        self.proj = nn.Conv2d(self.out_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 nb_cls=80,
                 img_size=[512, 32],
                 patch_size=[8, 32],
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 backbone='custom', # NEW ARGUMENT
                 use_cnn=True,
                 decoder_layers=0 # NEW ARGUMENT
                 **kwargs
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.layer_norm = LayerNorm()

        # --- BACKBONE SELECTION LOGIC ---
        if not use_cnn:
            # Identity mapper if no CNN is used (Linear Patch Projection could be added here if raw pixels)
            # For strict ablation "No CNN", we usually imply a simple Linear Patch Embedding
            # But for simplicity here, we assume x enters as features or we project it.
            # Here we use a simple Conv projection to act as "Linear Patch Embed"
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            # Note: This changes the effective grid size logic below depending on patch_size
            # For this code, we assume the input is still compatible.
        
        elif backbone == 'custom':
            # YOUR ORIGINAL CUSTOM RESNET
            self.patch_embed = resnet18.ResNet18(embed_dim)
        
        elif backbone in ['resnet18', 'resnet50', 'vgg16']:
            # STANDARD TORCHVISION BACKBONES
            self.patch_embed = TorchVisionAdapter(backbone, embed_dim)
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        # --- NEW: DECODER ABLATION SETUP ---
        self.decoder_layers_count = decoder_layers
        if self.decoder_layers_count > 0:
            # Define standard Transformer Decoder layer
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation='gelu',
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        else:
            self.decoder = None

        # --------------------------------------------------------------------------

        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, self.num_patches,
                  mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # pos_embed = get_2d_sincos_pos_embed(self.embed_dim, [1, self.nb_query])
        # self.qry_tokens.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:,idx:idx + max_span_length,:] = 0
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        x_masked = x * mask + (1 - mask) * self.mask_token
        return x_masked

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        # embed patches
        x = self.layer_norm(x)
        x = self.patch_embed(x)
        
        # Flatten logic: (B, C, H, W) -> (B, C, N) -> (B, N, C)
        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        
        # masking: length -> length * mask_ratio
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        
        # Add Positional Embedding (Check for size mismatch in case of different backbones)
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
             # Interpolate Pos Embed if sizes mismatch (e.g. different backbone strides)
             # This is a safety catch.
            pos_emb_resized = F.interpolate(
                self.pos_embed.permute(0, 2, 1), size=x.shape[1], mode='linear'
            ).permute(0, 2, 1)
            x = x + pos_emb_resized

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # --- NEW: DECODER PATH (Table 8 Ablation) ---
        if self.decoder is not None:
            # We use the encoder output 'x' as both the target and the memory
            # This refines global context before the CTC head
            x = self.decoder(tgt=x, memory=x)

        
        x = self.norm(x)
        # To CTC Loss
        x = self.head(x)
        x = self.layer_norm(x)

        return x


def create_model(nb_cls, img_size, backbone='custom', use_cnn=True, **kwargs):
    # --- FIX: POP ARGUMENTS FROM KWARGS ---
    # This prevents sending them twice (once explicitly below, and once inside **kwargs)
    decoder_layers = kwargs.pop('decoder_layers', 0)
    depth = kwargs.pop('depth', 4)
    heads = kwargs.pop('heads', 6) 
    
    # Clean up other potential duplicate keys if necessary
    kwargs.pop('num_heads', None) 

    model = MaskedAutoencoderViT(
        nb_cls=nb_cls,
        img_size=img_size,
        patch_size=(4, 64),
        embed_dim=768,
        depth=depth,
        num_heads=heads,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        backbone=backbone,
        use_cnn=use_cnn,
        decoder_layers=decoder_layers,
        **kwargs  # Now this is safe because we removed the conflicting keys above
    )
    
    return model
