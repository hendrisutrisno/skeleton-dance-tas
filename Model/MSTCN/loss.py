import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Optional
import math

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d tensor.
    Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """

    def __init__(self, kernel_size: int = 15, sigma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrid = torch.meshgrid(torch.arange(kernel_size), indexing='ij')[0].float()

        mean = (kernel_size - 1) / 2
        kernel = kernel / (sigma * math.sqrt(2 * math.pi))
        kernel = kernel * torch.exp(-(((meshgrid - mean) / sigma) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        # kernel = kernel / torch.max(kernel)

        self.kernel = kernel.view(1, 1, *kernel.size())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        _, c, _ = inputs.shape
        inputs = F.pad(
            inputs,
            pad=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            mode="reflect",
        )
        kernel = self.kernel.repeat(c, *[1] * (self.kernel.dim() - 1)).to(inputs.device)
        return F.conv1d(inputs, weight=kernel, groups=c)

class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None, 
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(
            preds, target, reduction="none", pos_weight= self.pos_weight
        )  # (N,)

        pt = torch.exp(-bce)

        if self.alpha is not None:
            alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)  # (N,)
            bce = alpha_t * bce

        loss = (1.0 - pt).pow(self.gamma) * bce

        return loss.sum() / target.shape[0]
        

class TMSE(nn.Module):
    def __init__(self, labels, threshold: float = 4, ignore_index: int = -100, weight: Optional[float] = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index= ignore_index)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, *arg, **kwargs):
        return self.loss(*arg)

    def loss(self, preds: torch.Tensor, mask: torch.Tensor, target) -> torch.Tensor:
        loss = 0.
        loss += self.ce(preds.transpose(2, 1).contiguous().view(-1, labels), target.view(-1, labels))
        loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(preds[:, :, 1:], dim=1), F.log_softmax(preds.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
        
        return loss

class GaussianSimilarityTMSE(nn.Module):
    def __init__(self, labels, ignore_index, threshold: float = 4, sigma: float = 1.0, weight: Optional[float] = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(weight= weight, ignore_index=ignore_index)
        self.seg_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.sigma = sigma
    
    def forward(self, *arg, **kwargs):
        return self.loss(*arg, **kwargs)
    
    def loss(
        self, preds: torch.Tensor, target, mask: torch.Tensor, sim_index: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            preds: the output of model before softmax. (N, C, T)
            gts: Ground Truth. (N, T)
            sim_index: similarity index. (N, C, T)
        Return:
            the value of Temporal MSE weighted by Gaussian Similarity.
        """
        total_loss = 0.
        loss = self.ce(preds.transpose(2, 1).contiguous().view(-1, labels), target.view(-1))
        sim = sim_index * mask[:, 0:1, :]
        # calculate gaussian similarity
        diff = sim[:, :, 1:] - sim[:, :, :-1]
        similarity = torch.exp(-torch.norm(diff, dim=1) / (2 * self.sigma ** 2))

        g_loss = torch.clamp(self.mse(F.log_softmax(preds[:, :, 1:], dim=1), F.log_softmax(preds.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:]
        g_loss = similarity * g_loss
        total_loss += loss + (0.15 * torch.mean(g_loss))

        return total_loss

class BoundaryRegressionLoss(nn.Module):
    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
    """

    def __init__(
        self,
        weight: Optional[float] = None,
        pos_weight: Optional[float] = None,
        facal = False
    ) -> None:
        super().__init__()
        if facal:
            self.bce = FocalLoss(weight=weight, pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
        self.smooth = GaussianSmoothing()

    def forward(self, *arg, **kwargs):
        return self.loss(*arg)

    def loss(self, preds: torch.Tensor, gts: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch. (N, 1, T).
            masks: torch.bool (N, 1, T).
        """

        loss = 0.0
        batch_size = preds.shape[0]
        gt = self.smooth(gts)
        bool_m =  masks[:, 0:1, :] != 0
        pred = preds[bool_m]
        gt = gt[bool_m]

        loss += self.bce(pred, gt)
        return loss / batch_size

class BoundaryheatmapLoss(nn.Module):
    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
    """

    def __init__(
        self
    ) -> None:
        super().__init__()
        self.bce = nn.KLDivLoss(reduction="batchmean")
        self.filter = GaussianSmoothing()
        self.sigmoid = nn.LogSigmoid()

    def forward(self, *arg, **kwargs):
        return self.loss(*arg)

    def loss(self, preds: torch.Tensor, gts: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch. (N, 1, T).
            masks: torch.bool (N, 1, T).
        """

        loss = 0.0
        bool_m =  masks[:, 0:1, :] != 0
        pred = preds[bool_m]
        gt = gts[bool_m]

        gt = self.filter(gt).squeeze(1)

        loss += self.bce(self.sigmoid(pred), self.sigmoid(gt))

        print(loss)
        return loss