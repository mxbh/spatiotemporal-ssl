# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .vit import (VisionTransformer, vit_tiny_patch2_32, vit_small_patch2_32, vit_small_patch16_224, 
                  vit_base_patch16_224, vit_base_patch16_96, vit_small_patch16_128)
from .fusion_vit import (EarlyFusionVisionTransformer, early_fusion_vit_small_patch16_128, early_fusion_vit_small_patch2_32,
                         early_fusion_vit_tiny_patch16_128, early_fusion_vit_small_patch16_224, early_fusion_vit_base_patch16_128, early_fusion_vit_large_patch16_128)

