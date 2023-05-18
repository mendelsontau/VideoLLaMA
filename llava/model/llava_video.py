#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from slowfast.slowfast.models import build_model
import slowfast.slowfast.utils.checkpoint as cu

from argparse import Namespace

import re


DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


class LlavaVideoConfig(LlamaConfig):
    model_type = "llavaVideo"


class LlavaVideoLlamaModel(LlamaModel):
    config_class = LlavaVideoConfig

    def __init__(self, config: LlamaConfig, video_config, mm_vision_tower=None, mm_hidden_size=None):
        super(LlavaVideoLlamaModel, self).__init__(config)

        self.video_config = video_config
        self.vision_config = Namespace()


    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):


        self.config.mm_vision_tower = vision_tower

        if not hasattr(self, 'vision_tower'):
            vision_tower = build_model(self.video_config)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        cu.load_test_checkpoint(self.video_config, self.vision_tower[0])

        vision_config = Namespace()

        #MViTv2-S
        self.vision_config.hidden_size = 768
        num_patches = 7 * 7 * 8 

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = self.vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(self.vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_token_len=num_patches,
            vision_config=self.vision_config
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        videos: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and videos is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(videos) is list:
                    # variable length images
                    image_features = []
                    for image in videos:
                        image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_features, cls = vision_tower(videos.unsqueeze(0))
            if type(videos) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(image_features.shape[1], self.vision_config.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if self.vision_config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == self.vision_config.im_start_token).sum() != (cur_input_ids == self.vision_config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == self.vision_config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != self.vision_config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlavaVideoLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LlavaVideoLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaVideoConfig

    def __init__(self, config, video_config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaVideoLlamaModel(config, video_config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.tokenizer = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        mode: str = "train",
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        videos: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        ssv2_label: Optional[torch.Tensor] = None,
        ssv2_video_idx: Optional[torch.Tensor] = None,
        ssv2_time: Optional[torch.Tensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            videos=videos
        )

        hidden_states = outputs[0]
        if mode == "train":
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                pred_tokens = torch.argmax(shift_logits,dim=2)
                shift_labels = labels[..., 1:].contiguous()
                gt_tokens = shift_labels
                gt_labels = []
                for j in range(gt_tokens.shape[0]):
                    gt_label = gt_tokens[j]
                    gt_label = gt_label.tolist()
                    gt_label = torch.tensor([idx for idx in gt_label if idx > 0 ])
                    gt_labels.append(gt_label)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                answers = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
                gt_answers = self.tokenizer.batch_decode(gt_labels, skip_special_tokens=True)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                with open("output_train.txt", "a") as file:
                    for a,b in zip(gt_answers,answers):
                        a_assistant = re.search("Assistant",a)
                        b_assistant = re.search("Assistant",b)
                        if b_assistant and a_assistant:
                            a = a[a_assistant.regs[0][0]:]
                            b = b[b_assistant.regs[0][0]:]
                            line = f"real answer: {a} \n predicted answer: {b} \n"
                            file.write(line)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            logits = self.lm_head(hidden_states)
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                pred_tokens = torch.argmax(shift_logits,dim=2)
                shift_labels = labels[..., 1:].contiguous()
                gt_tokens = shift_labels
                gt_labels = []
                for j in range(gt_tokens.shape[0]):
                    gt_label = gt_tokens[j]
                    gt_label = gt_label.tolist()
                    gt_label = torch.tensor([idx for idx in gt_label if idx > 0 ])
                    gt_labels.append(gt_label)
                # Flatten the tokens
                answers = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
                gt_answers = self.tokenizer.batch_decode(gt_labels, skip_special_tokens=True)
                with open("output_test.txt", "a") as file:
                    for a,b in zip(gt_answers,answers):
                        a_assistant = re.search("Assistant",a)
                        b_assistant = re.search("Assistant",b)
                        if b_assistant and a_assistant:
                            a = a[a_assistant.regs[0][0]:]
                            b = b[b_assistant.regs[0][0]:]
                        line = f"real answer: {a} \n predicted answer: {b} \n"
                        file.write(line)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device, vision_config,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):


        self.tokenizer = tokenizer                            
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert input_embeddings.shape == embed_tokens_weight.shape
                assert num_new_tokens == 2
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]

AutoConfig.register("llavaVideo", LlavaVideoConfig)
AutoModelForCausalLM.register(LlavaVideoConfig, LlavaVideoLlamaForCausalLM)
