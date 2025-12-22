# modules/modeling.py
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import torch
from torch import nn
import numpy as np

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer_Text, Transformer_Gate
from modules.module_clip import CLIP, convert_weights

logger = logging.getLogger(__name__)
allgather = AllGather.apply


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(
                target_name, target_attr_name, getattr(target_config, target_attr_name)
            ))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        task_config = kwargs.get("task_config", None)
        if task_config is not None:
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None:
            state_dict = {}

        pretrained_clip_name = getattr(task_config, 'pretrained_clip_name', "ViT-B/32")
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(
            cross_model_name, cache_dir, type_vocab_size,
            state_dict=None, task_config=task_config
        )
        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        if model.sim_header in ["seqLSTM", "seqTransf"]:
            contain_frame_position = any(key.find("frame_position_embeddings") > -1 for key in state_dict.keys())
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                        if num_layer < task_config.audio_query_layers:
                            state_dict[key.replace("transformer.", "transformer_Fusion.")] = val.clone()
                            continue

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model


class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
        self._stage_one = True
        self._stage_two = False
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        vit = "visual.proj" in clip_state_dict
        assert vit
        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in clip_state_dict.keys()
                             if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len({k.split(".")[2] for k in clip_state_dict.keys()
                                  if k.startswith("transformer.resblocks.")})

        self.linear_patch = getattr(task_config, "linear_patch", "2d")
        self.hidden_size = cross_config.hidden_size

        cut_top_layer = 0
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads,
            transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch
        ).float()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]
        convert_weights(self.clip)

        self.sim_header = getattr(self.task_config, "sim_header", "seqTransf")
        self.lambda_ = self.task_config.temperature

        num_query_token = 12
        self.query_tokens = nn.Parameter(torch.zeros(cross_config.hidden_size, num_query_token))
        nn.init.orthogonal_(self.query_tokens, 1.0)

        if self.loose_type is False:
            cross_config.max_position_embeddings = context_length
            cross_config = update_attr(
                "cross_config", cross_config, "num_hidden_layers",
                self.task_config, "cross_num_hidden_layers"
            )
            self.cross = CrossModel(cross_config)
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            self.query_position_embeddings = nn.Embedding(num_query_token, cross_config.hidden_size)

        if self.sim_header == "seqTransf":
            self.transformer_Fusion = Transformer_Text(
                width=transformer_width,
                layers=self.task_config.audio_query_layers,
                heads=transformer_heads
            )
            self.transformerClip = Transformer_Gate(
                width=transformer_width,
                layers=self.task_config.cross_num_hidden_layers,
                heads=transformer_heads
            )

            self.five_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 5, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 5),
                nn.Sigmoid()
            )

        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(
                input_size=cross_config.hidden_size,
                hidden_size=cross_config.hidden_size,
                batch_first=True, bidirectional=False, num_layers=1
            )

        self.loss_fct_vis = CrossEn()
        self.apply(self.init_weights)

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        denom = torch.sum(attention_mask_un, dim=1, dtype=torch.float).clamp(min=1e-6)
        text_out = torch.sum(sequence_output, dim=1) / denom
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _global_from_audio(self, a, how="cls"):
        if how == "cls":
            g = a[:, 0, :]
        else:
            g = a.mean(dim=1)
        return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def get_text_global(self, input_ids, token_type_ids, attention_mask):
        seq = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=False)
        text_global = self._mean_pooling_for_similarity_sequence(seq, attention_mask)
        valid = (attention_mask.sum(dim=1) > 0).float().unsqueeze(-1)
        text_global = text_global * valid
        norm = text_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        text_global = (text_global / norm) * valid
        return text_global

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        bs_pair = input_ids.size(0)
        if input_ids.shape[1] != 20:
            sequence_hidden = self.clip.encode_text(input_ids).float()
            sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        else:
            sequence_hidden = [self.clip.encode_text(input_ids[:, i, :]).float() for i in range(input_ids.shape[1])]
            sequence_hidden = torch.stack(sequence_hidden, dim=1)
        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        return visual_hidden

    def get_audio_output(self, audio):
        audio = audio.squeeze(1)
        audio_hidden = self.clip.encode_audio(audio).float()
        return audio_hidden

    def _fuse_av_only(self, visual_output, video_mask, audio_output):
        query_tokens = self.query_tokens.t().unsqueeze(0)
        query_embed = query_tokens.expand(visual_output.shape[0], -1, -1)

        position_ids = torch.arange(query_embed.size(1), dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        query_position_embeddings = self.query_position_embeddings(position_ids)
        query_embed = query_embed + query_position_embeddings

        query_mask = torch.ones(visual_output.shape[0], query_embed.shape[1], device=visual_output.device)
        extended_query_mask = (1.0 - query_mask.unsqueeze(-1)) * -1000000.0
        extended_query_mask = extended_query_mask.expand(-1, -1, audio_output.size(1))

        query_embed = query_embed.permute(1, 0, 2)
        audio_output = audio_output.permute(1, 0, 2)
        qa_output = self.transformer_Fusion(query_embed, audio_output, extended_query_mask)
        qa_output = qa_output.permute(1, 0, 2).contiguous()
        audio_output = audio_output.permute(1, 0, 2).contiguous()

        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)

        visual_output_original = visual_output
        visual_output = visual_output + frame_position_embeddings

        extended_video_mask = (1.0 - video_mask.unsqueeze(-1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, -1, qa_output.size(1))

        qa_output = qa_output.permute(1, 0, 2)
        visual_output = visual_output.permute(1, 0, 2)
        fusion_output, _, _, attn_gate_list, ff_gate_list = self.transformerClip(
            visual_output, qa_output, extended_video_mask
        )
        fusion_output = fusion_output.permute(1, 0, 2)

        fusion_output = 0.05 * fusion_output + 0.95 * visual_output_original
        fusion_output = fusion_output.contiguous()

        av_fused_output_global = self._mean_pooling_for_similarity_visual(fusion_output, video_mask)
        av_fused_output_global = av_fused_output_global / av_fused_output_global.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        attn_gate_tensor = torch.stack(attn_gate_list, 1) if isinstance(attn_gate_list, (list, tuple)) else None
        ff_gate_tensor = torch.stack(ff_gate_list, 1) if isinstance(ff_gate_list, (list, tuple)) else None
        return av_fused_output_global, attn_gate_tensor, ff_gate_tensor

    def _five_weighted(self, av_g, obj_g, act_g, att_g, audm_g):
        cat = torch.cat([av_g, obj_g, act_g, att_g, audm_g], dim=-1)
        w = self.five_gate(cat)
        comps = torch.stack([av_g, obj_g, act_g, att_g, audm_g], dim=1)
        q = torch.sum(w.unsqueeze(-1) * comps, dim=1)
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return q, w

    def encode_query(self, q_input_ids, q_token_type_ids, q_attention_mask, q_video, q_video_mask, q_audio):
        if q_video_mask.dim() == 3 and q_video_mask.size(1) == 1:
            q_video_mask = q_video_mask.squeeze(1)
        elif q_video_mask.dim() != 2:
            q_video_mask = q_video_mask.view(q_video_mask.size(0), -1)

        if q_input_ids.dim() != 3 or q_input_ids.size(1) < 4:
            raise ValueError("q_input_ids must be [B,4,L]")

        obj_g = self.get_text_global(q_input_ids[:, 0, :], q_token_type_ids[:, 0, :], q_attention_mask[:, 0, :])
        act_g = self.get_text_global(q_input_ids[:, 1, :], q_token_type_ids[:, 1, :], q_attention_mask[:, 1, :])
        att_g = self.get_text_global(q_input_ids[:, 2, :], q_token_type_ids[:, 2, :], q_attention_mask[:, 2, :])
        audm_g = self.get_text_global(q_input_ids[:, 3, :], q_token_type_ids[:, 3, :], q_attention_mask[:, 3, :])

        v_tok = self.get_visual_output(q_video, q_video_mask, shaped=False)
        a_tok = self.get_audio_output(q_audio)

        av_g, _, _ = self._fuse_av_only(v_tok, q_video_mask, a_tok)
        q, w = self._five_weighted(av_g, obj_g, act_g, att_g, audm_g)
        self._cached_five_weights = w
        return q

    def encode_target(self, t_video, t_video_mask, t_audio):
        if t_video_mask.dim() == 3 and t_video_mask.size(1) == 1:
            t_video_mask = t_video_mask.squeeze(1)
        elif t_video_mask.dim() != 2:
            t_video_mask = t_video_mask.view(t_video_mask.size(0), -1)

        v_tok = self.get_visual_output(t_video, t_video_mask, shaped=False)
        a_tok = self.get_audio_output(t_audio)

        t_av_global, _, _ = self._fuse_av_only(v_tok, t_video_mask, a_tok)
        return t_av_global

    def forward(self,
                q_input_ids, q_token_type_ids, q_attention_mask,
                q_video, q_video_mask, q_audio,
                t_video, t_video_mask, t_audio):
        assert self.sim_header == "seqTransf", "Gated fusion is required."

        q_global = self.encode_query(q_input_ids, q_token_type_ids, q_attention_mask, q_video, q_video_mask, q_audio)
        t_av_global = self.encode_target(t_video, t_video_mask, t_audio)
        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            q_all = allgather(q_global, self.task_config)
            t_all = allgather(t_av_global, self.task_config)
            torch.distributed.barrier()
            sim_matrix = torch.matmul(q_all, t_all.t()) * logit_scale
            loss = (self.loss_fct_vis(sim_matrix) + self.loss_fct_vis(sim_matrix.t())) / 2.0
            return loss
        else:
            sim_matrix = torch.matmul(q_global, t_av_global.t()) * logit_scale
            return sim_matrix, (None, None, None, None)
