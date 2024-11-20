#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import copy
import torch
import logging
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from typing import Union, Dict, List, Tuple, Optional
import numpy as np
import torchaudio

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.train_utils.device_funcs import to_device
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.search import Hypothesis
from funasr.models.paraformer.cif_predictor import mae_loss
from funasr.train_utils.device_funcs import force_gatherable
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank



@tables.register("model_classes", "ParaformerV2")
@tables.register("model_classes", "paraformerV2")
class ParaformerV2(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        decoder: str = None,
        decoder_conf: Optional[Dict] = None,
        ctc: str = None,
        ctc_conf: Optional[Dict] = None,
        ctc_weight: float = 0.5,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        # report_cer: bool = True,
        # report_wer: bool = True,
        # sym_space: str = "<space>",
        # sym_blank: str = "<blank>",
        # extract_feats_in_collect_stats: bool = True,
        # predictor=None,
        predictor_weight: float = 0.0,
        predictor_bias: int = 0,
        sampling_ratio: float = 0.2,
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        use_1st_decoder_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if decoder is not None:
            decoder_class = tables.decoder_classes.get(decoder)
            decoder = decoder_class(
                vocab_size=vocab_size,
                encoder_output_size=encoder_output_size,
                **decoder_conf,
            )
        if ctc_weight > 0.0:

            if ctc_conf is None:
                ctc_conf = {}

            ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)


        # note that eos is the same as sos (equivalent ID)
        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1

        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            self.token2id = tokenizer.token2id
            self.blank_id = self.token2id.get("<blank>", blank_id)
            self.sos = self.token2id.get("<s>", self.sos)
            self.eos = self.token2id.get("</s>", self.eos)

            if hasattr(tokenizer, 'add_special_token_list'):
                add_special_token_list = tokenizer.add_special_token_list
            else:
                add_special_token_list = False
            if add_special_token_list:
                self.start_id_of_special_tokens = len(self.token2id) - len(add_special_token_list)
        else:
            self.token2id = None

        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        # self.token_list = token_list.copy()
        #
        # self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize

        self.encoder = encoder
        #
        # if not hasattr(self.encoder, "interctc_use_conditioning"):
        #     self.encoder.interctc_use_conditioning = False
        # if self.encoder.interctc_use_conditioning:
        #     self.encoder.conditioning_layer = torch.nn.Linear(
        #         vocab_size, self.encoder.output_size()
        #     )
        #
        # self.error_calculator = None
        #
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        #
        # if report_cer or report_wer:
        #     self.error_calculator = ErrorCalculator(
        #         token_list, sym_space, sym_blank, report_cer, report_wer
        #     )
        #
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        #
        # self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.share_embedding = share_embedding
        if self.share_embedding:
            self.decoder.embed = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.error_calculator = None

        self.total_token_num, self.error_num = 1, 0

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_ctc, cer_ctc = None, None
        stats = dict()

        # decoder: CTC branch
        loss_ctc, cer_ctc = self._calc_ctc_loss(encoder_out, encoder_out_lens, text, text_lengths)
        # Collect CTC branch stats
        stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
        stats["cer_ctc"] = cer_ctc

        # decoder: Attention decoder branch
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(encoder_out, encoder_out_lens, text, text_lengths)

        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        # loss = loss_ctc

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = (text_lengths + self.predictor_bias).sum()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens



    def calc_predictor(self, encoder_out, encoder_out_lens):

        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(
            encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id
        )
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index

    def cal_decoder_with_predictor(
        self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
    ):

        decoder_outs = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        decoder_out = decoder_outs[0]
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        return decoder_out, ys_pad_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):

        batch_size = encoder_out.size(0)
        with torch.no_grad():
            compressed_ctc_batch  = []
            ctc_probs = self.ctc.log_softmax(encoder_out).detach()

            for b in range(batch_size):
                ctc_prob = ctc_probs[b][: encoder_out_lens[b]].cpu() # [T, N]
                text_b = ys_pad[b][: ys_pad_lens[b]].cpu() # [1, U]
                text_audio_alignment = self.ctc.force_align(ctc_prob, text_b)
                text_audio_alignment = torch.tensor(text_audio_alignment)
                audio_text = self.ctc.remove_duplicates_and_blank(text_audio_alignment, self.blank_id)
                if len(audio_text) != ys_pad_lens[b]:
                    print (f"ctc alignment error: {audio_text}, {text_b}")
                # 把相同的不为0的帧的概率平均
                ctc_comp = self.average_repeats(ctc_prob, text_audio_alignment)
                if ctc_comp.size(0) != ys_pad_lens[b]:
                    print (f"ctc_comp error: {ctc_comp.size(0)}, {text_b}")
                compressed_ctc_batch.append(ctc_comp)

            padded_ctc_batch = pad_sequence(compressed_ctc_batch, batch_first=True).to(encoder_out.device)

        decoder_outs = self.decoder(encoder_out, encoder_out_lens, padded_ctc_batch, ys_pad_lens)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]


        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(decoder_out.view(-1, self.vocab_size),  ys_pad, ignore_label=self.ignore_id,)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def average_repeats(self, ctc_prob, alignment):
        """
        Averages the repeated frames based on alignment.

        Args:
            ctc_prob (torch.Tensor): Tensor of shape [T, VocabSize + 1] representing frame-wise CTC posteriors.
            alignment (torch.Tensor): Tensor of shape [T,] representing the target alignment from Viterbi algorithm.

        Returns:
            torch.Tensor: Compressed CTC posterior with repeated frames averaged and blanks removed.
        """
        unique_tokens = []
        unique_probs = []
        current_sum = None
        current_count = 0

        for t in range(alignment.size(0)):
            token = alignment[t].item()
            prob = ctc_prob[t]

            if len(unique_tokens) == 0 or token != unique_tokens[-1]:
                if current_count > 0:
                    unique_probs.append(current_sum / current_count)
                unique_tokens.append(token)
                current_sum = prob
                current_count = 1
            else:
                current_sum += prob
                current_count += 1

        # Append the last averaged probability
        if current_count > 0:
            unique_probs.append(current_sum / current_count)

        non_blank_ctc_prob = []
        responded_id = self.ctc.remove_duplicates(alignment)
        for i in range(len(responded_id)):
            id  = responded_id[i]
            if id == self.blank_id:
                continue
            else:
                non_blank_ctc_prob.append(unique_probs[i])

        compressed_ctc_prob = torch.stack(non_blank_ctc_prob, dim=0)
        return compressed_ctc_prob



    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def init_beam_search(
        self,
        **kwargs,
    ):
        from funasr.models.paraformer.search import BeamSearchPara
        from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        if self.ctc != None:
            ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos)
            scorers.update(ctc=ctc)
        token_list = kwargs.get("token_list")
        scorers.update(
            length_bonus=LengthBonus(len(token_list)),
        )

        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        weights = dict(
            decoder=1.0 - kwargs.get("decoding_ctc_weight"),
            ctc=kwargs.get("decoding_ctc_weight", 0.0),
            lm=kwargs.get("lm_weight", 0.0),
            ngram=kwargs.get("ngram_weight", 0.0),
            length_bonus=kwargs.get("penalty", 0.0),
        )
        beam_search = BeamSearchPara(
            beam_size=kwargs.get("beam_size", 2),
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=None if self.ctc_weight == 1.0 else "full",
        )
        # beam_search.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        # for scorer in scorers.values():
        #     if isinstance(scorer, torch.nn.Module):
        #         scorer.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        self.beam_search = beam_search

    '''FunASR inference'''
    # def inference(
    #     self,
    #     data_in,
    #     data_lengths=None,
    #     key: list = None,
    #     tokenizer=None,
    #     frontend=None,
    #     **kwargs,
    # ):
    #     # init beamsearch
    #     is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc != None
    #     is_use_lm = (
    #         kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
    #     )
    #     pred_timestamp = kwargs.get("pred_timestamp", False)
    #     if self.beam_search is None and (is_use_lm or is_use_ctc):
    #         logging.info("enable beam_search")
    #         self.init_beam_search(**kwargs)
    #         self.nbest = kwargs.get("nbest", 1)
    #
    #     meta_data = {}
    #     if (
    #         isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
    #     ):  # fbank
    #         speech, speech_lengths = data_in, data_lengths
    #         if len(speech.shape) < 3:
    #             speech = speech[None, :, :]
    #         if speech_lengths is not None:
    #             speech_lengths = speech_lengths.squeeze(-1)
    #         else:
    #             speech_lengths = speech.shape[1]
    #     else:
    #         # extract fbank feats
    #         time1 = time.perf_counter()
    #         audio_sample_list = load_audio_text_image_video(
    #             data_in,
    #             fs=frontend.fs,
    #             audio_fs=kwargs.get("fs", 16000),
    #             data_type=kwargs.get("data_type", "sound"),
    #             tokenizer=tokenizer,
    #         )
    #         time2 = time.perf_counter()
    #         meta_data["load_data"] = f"{time2 - time1:0.3f}"
    #         speech, speech_lengths = extract_fbank(
    #             audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
    #         )
    #         time3 = time.perf_counter()
    #         meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
    #         meta_data["batch_data_time"] = (
    #             speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
    #         )
    #
    #     speech = speech.to(device=kwargs["device"])
    #     speech_lengths = speech_lengths.to(device=kwargs["device"])
    #     # Encoder
    #     if kwargs.get("fp16", False):
    #         speech = speech.half()
    #     encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
    #     if isinstance(encoder_out, tuple):
    #         encoder_out = encoder_out[0]
    #
    #     # predictor
    #     predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
    #     pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (
    #         predictor_outs[0],
    #         predictor_outs[1],
    #         predictor_outs[2],
    #         predictor_outs[3],
    #     )
    #
    #     pre_token_length = pre_token_length.round().long()
    #     if torch.max(pre_token_length) < 1:
    #         return []
    #     decoder_outs = self.cal_decoder_with_predictor(
    #         encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length
    #     )
    #     decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
    #
    #     results = []
    #     b, n, d = decoder_out.size()
    #     if isinstance(key[0], (list, tuple)):
    #         key = key[0]
    #     if len(key) < b:
    #         key = key * b
    #     for i in range(b):
    #         x = encoder_out[i, : encoder_out_lens[i], :]
    #         am_scores = decoder_out[i, : pre_token_length[i], :]
    #         if self.beam_search is not None:
    #             nbest_hyps = self.beam_search(
    #                 x=x,
    #                 am_scores=am_scores,
    #                 maxlenratio=kwargs.get("maxlenratio", 0.0),
    #                 minlenratio=kwargs.get("minlenratio", 0.0),
    #             )
    #
    #             nbest_hyps = nbest_hyps[: self.nbest]
    #         else:
    #
    #             yseq = am_scores.argmax(dim=-1)
    #             score = am_scores.max(dim=-1)[0]
    #             score = torch.sum(score, dim=-1)
    #             # pad with mask tokens to ensure compatibility with sos/eos tokens
    #             yseq = torch.tensor([self.sos] + yseq.tolist() + [self.eos], device=yseq.device)
    #             nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
    #         for nbest_idx, hyp in enumerate(nbest_hyps):
    #             ibest_writer = None
    #             if kwargs.get("output_dir") is not None:
    #                 if not hasattr(self, "writer"):
    #                     self.writer = DatadirWriter(kwargs.get("output_dir"))
    #                 ibest_writer = self.writer[f"{nbest_idx+1}best_recog"]
    #             # remove sos/eos and get results
    #             last_pos = -1
    #             if isinstance(hyp.yseq, list):
    #                 token_int = hyp.yseq[1:last_pos]
    #             else:
    #                 token_int = hyp.yseq[1:last_pos].tolist()
    #
    #             # remove blank symbol id, which is assumed to be 0
    #             token_int = list(
    #                 filter(
    #                     lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
    #                 )
    #             )
    #
    #             if tokenizer is not None:
    #                 # Change integer-ids to tokens
    #                 token = tokenizer.ids2tokens(token_int)
    #                 text_postprocessed = tokenizer.tokens2text(token)
    #
    #                 if pred_timestamp:
    #                     timestamp_str, timestamp = ts_prediction_lfr6_standard(
    #                         pre_peak_index[i],
    #                         alphas[i],
    #                         copy.copy(token),
    #                         vad_offset=kwargs.get("begin_time", 0),
    #                         upsample_rate=1,
    #                     )
    #                     if not hasattr(tokenizer, "bpemodel"):
    #                         text_postprocessed, time_stamp_postprocessed, _ = postprocess_utils.sentence_postprocess(token, timestamp)
    #                     result_i = {"key": key[i], "text": text_postprocessed, "timestamp": time_stamp_postprocessed,}
    #                 else:
    #                     if not hasattr(tokenizer, "bpemodel"):
    #                         text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
    #                     result_i = {"key": key[i], "text": text_postprocessed}
    #
    #                 if ibest_writer is not None:
    #                     ibest_writer["token"][key[i]] = " ".join(token)
    #                     # ibest_writer["text"][key[i]] = text
    #                     ibest_writer["text"][key[i]] = text_postprocessed
    #             else:
    #                 result_i = {"key": key[i], "token_int": token_int}
    #             results.append(result_i)
    #
    #     return results, meta_data




    def cal_topk_error_rate(self, label, top):
        label_array = np.array(label)
        top_array = np.array(top)
        # 计算每个位置上，是否有至少一个匹配的预测值
        correct_at_least_one = np.any(top_array == label_array, axis=0)
        # 计算每个位置是否错误（没有任何预测值匹配标签）
        error_positions = ~correct_at_least_one
        return np.sum(error_positions)

