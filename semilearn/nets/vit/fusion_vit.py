import torch
import torch.nn as nn
from semilearn.nets.utils import load_checkpoint
from .vit import VisionTransformer, Mlp, PatchEmbed, Block


class EarlyFusionVisionTransformer(VisionTransformer):
    """Adds side information through an additional token that is inserted into the list of tokens,
    i.e., (cls_tkn, patch_tkn_0_0, ..., patch_tkn_p_p, meta_tkn)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        metainfo_in_features=4,
        metainfo_dropout=0.0,
        always_meta_token=None
    ):
        super(EarlyFusionVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
        )
        # fix positional embeddings and add meta tkn embedding 
        self.metainfo_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.embed_dim) * .02 # https://github.com/huggingface/pytorch-image-models/blob/7f19a4cce7004eee11704956d89c94566424f5ee/timm/models/vision_transformer.py#L493
        )
        self.metainfo_in_features = metainfo_in_features
        self.metainfo_dropout = metainfo_dropout 
        self.always_meta_token = always_meta_token

    def extract(self, x, metainfo_embed=None):
        x = self.patch_embed(x) # [B,N,C]
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), 
                       x), 
                      dim=1) # [B,1+N,C]
        x = self.pos_drop(x + self.pos_embed)
        if metainfo_embed is not None:
            x = torch.cat((x, metainfo_embed.unsqueeze(1) + self.metainfo_pos_embed), dim=1) # [B,1+N+1,C]
        else:
            assert self.always_meta_token is not None
            if self.always_meta_token:
                x = torch.cat((x, self.metainfo_pos_embed.expand(x.shape[0], -1, -1)), dim=1) # [B,1+N+1,C]
            else:
                pass # only use class and patch tokens
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x, metainfo_embed=None, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            return self.head(x)

        x = self.extract(x, metainfo_embed)
        if self.global_pool:
            cls_tkn = x[:, 1:-1].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        else:
            cls_tkn = x
        cls_tkn = self.fc_norm(cls_tkn)

        if only_feat:
            return cls_tkn

        output = self.head(cls_tkn)
        result_dict = {"logits": output, "feat": cls_tkn}
        return result_dict


def early_fusion_vit_small_patch16_128(
        metainfo_in_features=4, 
        pretrained=False, 
        pretrained_path=None, 
        **kwargs
    ):
    """ViT-Small (ViT-S/16)"""
    model_kwargs = dict(
        img_size=128,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=0.2,
        **kwargs
    )
    model = EarlyFusionVisionTransformer(
        **model_kwargs, 
        metainfo_in_features=metainfo_in_features
    )
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def early_fusion_vit_small_patch16_224(
        metainfo_in_features=4, 
        pretrained=False, 
        pretrained_path=None, 
        **kwargs
    ):
    """ViT-Small (ViT-S/16)"""
    model_kwargs = dict(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=0.2,
        **kwargs
    )
    model = EarlyFusionVisionTransformer(
        **model_kwargs, 
        metainfo_in_features=metainfo_in_features
    )
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def early_fusion_vit_base_patch16_128(
        metainfo_in_features=4,
        pretrained=False, 
        pretrained_path=None, 
        **kwargs
    ):
    """ViT-Base (ViT-B/16)"""
    model_kwargs = dict(
        img_size=128,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.2,
        **kwargs
    )
    model = EarlyFusionVisionTransformer(
        **model_kwargs, 
        metainfo_in_features=metainfo_in_features
    )
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def early_fusion_vit_tiny_patch16_128(
        metainfo_in_features=4,
        pretrained=False, 
        pretrained_path=None, 
        **kwargs
    ):
    """ViT-Tiny (ViT-T/16)"""
    model_kwargs = dict(
        img_size=128,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        drop_path_rate=0.2,
        **kwargs
    )
    model = EarlyFusionVisionTransformer(
        **model_kwargs, 
        metainfo_in_features=metainfo_in_features
    )
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def early_fusion_vit_large_patch16_128(
        metainfo_in_features=4,
        pretrained=False, 
        pretrained_path=None, 
        **kwargs
    ):
    """ViT-Large (ViT-L/16)"""
    model_kwargs = dict(
        img_size=128,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=10,
        drop_path_rate=0.2,
        **kwargs
    )
    model = EarlyFusionVisionTransformer(
        **model_kwargs, 
        metainfo_in_features=metainfo_in_features
    )
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def early_fusion_vit_small_patch2_32(
        metainfo_in_features=4, 
        pretrained=False, 
        pretrained_path=None, 
        **kwargs
    ):
    """ViT-Small (ViT-S/2)"""
    model_kwargs = dict(
        img_size=32,
        patch_size=2,
        embed_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=0.2,
        **kwargs
    )
    model = EarlyFusionVisionTransformer(
        **model_kwargs, 
        metainfo_in_features=metainfo_in_features
    )
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
