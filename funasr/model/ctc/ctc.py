import logging

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import numpy as np

class CTC(nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or warpctc
        reduce: reduce the CTC loss into a scalar
    """

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = True,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        self.ignore_nan_grad = ignore_nan_grad

        if self.ctc_type == "builtin":
            self.ctc_loss = nn.CTCLoss(reduction="none")

        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc
            if ignore_nan_grad:
                logging.warning("ignore_nan_grad option is not supported for warp_ctc")
            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)

        elif self.ctc_type == "wenetctc":
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = nn.CTCLoss(blank=0, reduction=reduction_type, zero_infinity=True)

        else:
            raise ValueError(f'ctc_type must be "builtin" or "warpctc": {self.ctc_type}')

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin":
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                # ctc_grad: (L, B, O)
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    # Return as is
                    logging.warning("All samples in this mini-batch got nan grad. Returning nan value instead of CTC loss")
                elif size != th_pred.size(1):
                    logging.warning(f"{th_pred.size(1) - size}/{th_pred.size(1)} samples got nan grad. These were ignored for CTC loss.")

                    # Create mask for target
                    target_mask = torch.full([th_target.size(0)], 1, dtype=torch.bool,  device=th_target.device, )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    # Calc loss again using maksed data
                    loss = self.ctc_loss(th_pred[:, indices, :], th_target[target_mask], th_ilen[indices], th_olen[indices],)
            else:
                size = th_pred.size(1)

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "warpctc":
            # warpctc only supports float32
            th_pred = th_pred.to(dtype=torch.float32)

            th_target = th_target.cpu().int()
            th_ilen = th_ilen.cpu().int()
            th_olen = th_olen.cpu().int()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.reduce:
                # NOTE: sum() is needed to keep consistency since warpctc
                # return as tensor w/ shape (1,)
                # but builtin return as tensor w/o shape (scalar).
                loss = loss.sum()
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, h_lens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        if self.ctc_type == "wenetctc":
            ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            ys_hat = ys_hat.log_softmax(2)
            loss = self.ctc_loss(ys_hat, ys_pad, h_lens, ys_lens)
            # Batch-size average
            loss = loss / ys_hat.size(1)
            ys_hat = ys_hat.transpose(0, 1)
        else:
            ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

            if self.ctc_type == "gtnctc":
                # gtn expects list form for ys
                ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
            else:
                # ys_hat: (B, L, D) -> (L, B, D)
                ys_hat = ys_hat.transpose(0, 1)
                # (B, L) -> (BxL,)
                ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

            loss = self.loss_fn(ys_hat, ys_true, h_lens, ys_lens).to(device=hs_pad.device, dtype=hs_pad.dtype)

        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def ctc_logprobs(self, hs_pad, blank_id=0, blank_penalty: float = 0.0,):
        """log softmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        if blank_penalty > 0.0:
            logits = self.ctc_lo(hs_pad)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.log_softmax(hs_pad)
        return ctc_probs

    def insert_blank(self, label, blank_id=0):
        """Insert blank token between every two label token."""
        label = np.expand_dims(label, 1)
        blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
        label = np.concatenate([blanks, label], axis=1)
        label = label.reshape(-1)
        label = np.append(label, label[0])
        return label

    def force_align(self, ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list:
        """ctc forced alignment.

        Args:
            torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
            torch.Tensor y: id sequence tensor 1d tensor (L)
            int blank_id: blank symbol index
        Returns:
            torch.Tensor: alignment result
        """
        ctc_probs = ctc_probs[None].cpu()
        y = y[None].cpu()
        alignments, _ = torchaudio.functional.forced_align(ctc_probs, y, blank=blank_id)
        return alignments[0]

    def remove_duplicates_and_blank(self, alignment, blank_id=0):
        """
        去除对齐路径中的空白标签和重复标签。

        alignment: 对齐路径，可能包含空白标签和重复标签。
        blank_id: 空白标签的 ID。

        返回：
        filtered_alignment: 去除空白和重复标签后的对齐路径。
        """
        filtered_alignment = []
        prev_token = None
        for token in alignment:
            if token != blank_id and token != prev_token:
                filtered_alignment.append(token)
            prev_token = token
        return filtered_alignment

    def remove_duplicates(self, alignment):
        """
        去除对齐路径中的空白标签和重复标签。

        alignment: 对齐路径，可能包含空白标签和重复标签。
        blank_id: 空白标签的 ID。

        返回：
        filtered_alignment: 去除空白和重复标签后的对齐路径。
        """
        filtered_alignment = []
        prev_token = None
        for token in alignment:
            if  token != prev_token:
                filtered_alignment.append(token)
            prev_token = token
        return filtered_alignment
