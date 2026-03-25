import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMaskedRecovery(nn.Module):
    """
    Feature Masked Recovery (FMR) Module
    Input:
        F : [B, C, H, W]  (high-level feature map from backbone)
    Output:
        F_fmr : [B, C, H, W]
        F_rec : recovered feature (for recovery loss)
        mask  : upsampled binary mask
    """

    def __init__(
        self,
        in_channels,
        block_size=4,          # s
        keep_prob=0.5          # p
    ):
        super().__init__()

        self.block_size = block_size
        self.keep_prob = keep_prob

        # -------- Recovery Head R(.) --------
        self.recovery_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # -------- Gated Fusion --------
        # input: concat(F, F_rec, |F-F_rec|)
        self.gate_conv = nn.Conv2d(
            in_channels * 3,
            in_channels,
            kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

        # -------- 添加分类头 ----------
        # 将特征图降维到 num_classes
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 2, kernel_size=1)
        )

    def generate_block_mask(self, x):
        """
        Generate structured Bernoulli mask
        """
        B, C, H, W = x.shape
        b = self.block_size

        grid_h = H // b
        grid_w = W // b

        # Bernoulli sampling
        mask = torch.bernoulli(
            torch.full((B, 1, grid_h, grid_w),
                       self.keep_prob,
                       device=x.device)
        )

        # nearest neighbor upsample
        mask = F.interpolate(mask, size=(H, W), mode="nearest")

        return mask

    def forward(self, F_in):
        """
        F_in: [B, C, H, W]
        """

        # Structured Masking
        mask = self.generate_block_mask(F_in)
        F_masked = F_in * mask

        # Feature Recovery
        F_rec = self.recovery_head(F_masked)

        # Gated Fusion
        diff = torch.abs(F_in - F_rec)

        fusion_input = torch.cat([F_in, F_rec, diff], dim=1)
        gate = self.sigmoid(self.gate_conv(fusion_input))

        F_fmr = gate * F_in + (1 - gate) * F_rec
        fmr_logits = self.classifier(F_fmr)
        rec_logits = self.classifier(F_rec)

        return F_fmr, F_rec, mask, fmr_logits, rec_logits