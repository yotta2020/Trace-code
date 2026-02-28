import torch
import torch.nn as nn
from ...utils import logger
from ..victim import BaseVictim
from typing import *
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import tempfile
import os
import sys

# Import CodeBLEU evaluator
try:
    project_root = Path(__file__).resolve().parents[6]
    # 指向包含 CodeT5 文件夹的父目录
    refine_base_path = project_root / "src" / "training" / "victim_model" / "CodeRefinement"
    if str(refine_base_path) not in sys.path:
        sys.path.insert(0, str(refine_base_path))
    
    # 使用完整的包路径导入
    from CodeT5.evaluator.CodeBLEU import calc_code_bleu
    CODEBLEU_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CodeRefinement CodeBLEU not available: {e}")
    CODEBLEU_AVAILABLE = False

MODEL_CLASSES = {"roberta_refine": (RobertaConfig, RobertaModel, RobertaTokenizer)}


class Seq2Seq(nn.Module):
    """
    Build Seqence-to-Sequence.

    Parameters:

    * `encoder`- encoder of seq2seq model. e.g. roberta
    * `decoder`- decoder of seq2seq model. e.g. transformer
    * `config`- configuration of encoder model.
    * `beam_size`- beam size for beam search.
    * `max_length`- max length of target for beam search.
    * `sos_id`- start of symbol ids in target for beam search.
    * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(
        self,
        encoder,
        decoder,
        config,
        beam_size=None,
        max_length=None,
        sos_id=None,
        eos_id=None,
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """Tie or clone module weights depending of weither we are using TorchScript or not"""
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(
            self.lm_head, self.encoder.embeddings.word_embeddings
        )

    def hidden_states(
        self, source_ids: torch.Tensor = None, source_mask: torch.Tensor = None
    ) -> List[torch.Tensor]:

        outputs = self.encoder(
            source_ids, attention_mask=source_mask, output_hidden_states=True
        )
        return outputs.hidden_states

    def forward(
        self, source_ids=None, source_mask=None, target_ids=None, target_mask=None
    ):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        if target_ids is not None:
            attn_mask = -1e4 * (
                1 - self.bias[: target_ids.shape[1], : target_ids.shape[1]]
            )
            tgt_embeddings = (
                self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            )
            out = self.decoder(
                tgt_embeddings,
                encoder_output,
                tgt_mask=attn_mask,
                memory_key_padding_mask=(1 - source_mask).bool(),
            )
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            # Reshape to batch dimension
            loss = loss.view(target_ids.shape[0], -1)
            # Only compute loss on active (non-padding) tokens
            loss = loss * active_loss.view(target_ids.shape[0], -1).float()
            # Average over active tokens for each sample
            loss = loss.sum(dim=1) / active_loss.view(target_ids.shape[0], -1).float().sum(dim=1)
            return loss
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i : i + 1]
                context_mask = source_mask[i : i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (
                        1 - self.bias[: input_ids.shape[1], : input_ids.shape[1]]
                    )
                    tgt_embeddings = (
                        self.encoder.embeddings(input_ids)
                        .permute([1, 0, 2])
                        .contiguous()
                    )
                    out = self.decoder(
                        tgt_embeddings,
                        context,
                        tgt_mask=attn_mask,
                        memory_key_padding_mask=(1 - context_mask).bool(),
                    )
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(
                        input_ids.data.index_select(0, beam.getCurrentOrigin())
                    )
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[: self.beam_size]
                pred = [
                    torch.cat(
                        [x.view(-1) for x in p] + [zero] * (self.max_length - len(p))
                    ).view(1, -1)
                    for p in pred
                ]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[: self.size - len(self.finished)]
        return self.finished[: self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


class Victim(BaseVictim):
    def __init__(self, cfg):
        super().__init__(cfg)
        logger.info(f"load model from {cfg.victim.model_path}")
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.victim.device == "gpu"
            else "cpu"
        )
        config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta_refine"]
        config = config_class.from_pretrained(cfg.victim.base_path, cache_dir=None)
        tokenizer = tokenizer_class.from_pretrained(
            cfg.victim.base_path, cache_dir=None
        )
        encoder = model_class.from_pretrained(cfg.victim.base_path, config=config)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            config=config,
            beam_size=5,
            max_length=256,
            sos_id=tokenizer.cls_token_id,
            eos_id=tokenizer.sep_token_id,
        )

        block_size = 256
        input_path = Path(cfg.victim.model_path)
        if input_path.is_file():
            model_path = input_path
        else:
            matches = list(input_path.glob("*model.bin"))
            if not matches:
                raise FileNotFoundError(f"在路径 {cfg.victim.model_path} 下找不到以 model.bin 结尾的模型文件")
            model_path = matches[0]
        model.load_state_dict(torch.load(model_path), strict=False)
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.device = device

    def process(self, js, use_target_label=False):
        tokenizer = self.tokenizer

        if use_target_label:
            target_code = js["target_label"]
        else:
            target_code = js["fixed"]

        source_code = js["buggy"]
        source_tokens = tokenizer.tokenize(source_code)[: self.block_size - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = self.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        target_tokens = tokenizer.tokenize(target_code)[: self.block_size - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = self.block_size - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        return source_ids, source_mask, target_ids, target_mask

    def getLosses(
        self, objs: List[dict], target_label: int = None, batch_size: int = 32
    ) -> List[float]:
        """Get per-sample losses for refinement task"""
        dataloader = self.refineDataLoader(objs, use_tqdm=False, batch_size=batch_size)
        losses = []

        for batch in tqdm(dataloader, ncols=100, desc="getLosses"):
            source_ids, source_mask, target_ids, target_mask = [
                item.to(self.device) for item in batch
            ]

            with torch.no_grad():
                # Model returns per-sample loss
                _losses = self.model(
                    source_ids=source_ids,
                    source_mask=source_mask,
                    target_ids=target_ids,
                    target_mask=target_mask,
                )
                losses.extend(_losses.tolist())

        return losses

    def refineDataLoader(
        self, objs: List[dict], use_tqdm: bool = True, batch_size: int = 32
    ) -> DataLoader:
        """DataLoader specifically for refinement task with source and target"""
        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc=f"process")
        else:
            pbar = objs

        source_ids_list = []
        source_mask_list = []
        target_ids_list = []
        target_mask_list = []

        for obj in pbar:
            source_ids, source_mask, target_ids, target_mask = self.process(obj)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            source_mask_list.append(torch.tensor(source_mask, dtype=torch.long))
            target_ids_list.append(torch.tensor(target_ids, dtype=torch.long))
            target_mask_list.append(torch.tensor(target_mask, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_source_mask = torch.stack(source_mask_list)
        all_target_ids = torch.stack(target_ids_list)
        all_target_mask = torch.stack(target_mask_list)

        tensor_dataset = TensorDataset(
            all_source_ids, all_source_mask, all_target_ids, all_target_mask
        )
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(
            tensor_dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        return dataloader

    def dataLoader(
        self, objs: List[dict], use_tqdm: bool = True, batch_size: int = 32
    ) -> DataLoader:
        """DataLoader for hidden state extraction (source only)"""
        if use_tqdm:
            pbar = tqdm(objs, ncols=100, desc=f"process")
        else:
            pbar = objs

        source_ids_list = []
        source_mask_list = []

        for obj in pbar:
            source_ids, source_mask, _, _ = self.process(obj)
            source_ids_list.append(torch.tensor(source_ids, dtype=torch.long))
            source_mask_list.append(torch.tensor(source_mask, dtype=torch.long))

        all_source_ids = torch.stack(source_ids_list)
        all_source_mask = torch.stack(source_mask_list)
        tensor_dataset = TensorDataset(all_source_ids, all_source_mask)
        sampler = SequentialSampler(tensor_dataset)
        dataloader = DataLoader(
            tensor_dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        return dataloader

    def getLastLayerCLSHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerCLSHiddenState"
        ):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                hidden_state = self.model.hidden_states(
                    source_ids=source_ids, source_mask=source_mask
                )
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # (batch_size, hidden_dim)
                hidden_state = hidden_state[:, -1, 0, :].squeeze(1).squeeze(1)
                # print(f"hidden_state = {hidden_state.shape}")
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getLastLayerHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader, ncols=100, desc="getLastLayerHiddenState"
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # (batch_size, seq_len, hidden_dim)
                hidden_state = hidden_state[:, -1, :, :].squeeze()
                t_batch_size = hidden_state.shape[0]
                hidden_state = hidden_state.view(t_batch_size, -1)

                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getLastLayerMaxPoolingHiddenState(
        self, objs: List[dict], to_numpy: bool = False, batch_size: int = 32
    ) -> List[np.ndarray]:

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        hidden_states = []
        for batch in tqdm(
            dataloader,
            ncols=100,
            desc="getLastLayerMaxPoolingHiddenState"
        ):
            inputs = batch[0].to(self.device)
            with torch.no_grad():
                hidden_state = self.model.hidden_states(inputs)
                # (batch_size, layer_num, seq_len, hidden_dim)
                hidden_state = torch.stack(hidden_state).permute(1, 0, 2, 3)
                # (batch_size, hidden_dim)
                hidden_state = (
                    hidden_state[:, -1, :, :].squeeze().max(dim=1)[0].squeeze()
                )
                # cls_hidden_state = hidden_state[:, -1, 0, :].squeeze()
                # hidden_state = torch.cat((max_hidden_state, cls_hidden_state), dim=1)
                if to_numpy:
                    hidden_state = hidden_state.cpu().numpy()
                hidden_states.extend([h for h in hidden_state])
                torch.cuda.empty_cache()

        # (hidden_dim) * sample_num
        return hidden_states

    def getHiddenStatesNoBatch(self, objs: List[dict]) -> List[torch.Tensor]:

        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=len(objs))
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                hidden_state = self.model.hidden_states(source_ids, source_mask)
                return hidden_state

    def test(self, objs, use_tqdm=True, batch_size=32, target_label=None):
        """
        Evaluate model on Code Refinement task.

        Args:
            objs: List of data objects with 'buggy' and 'fixed' fields
            use_tqdm: Whether to show progress bar
            batch_size: Batch size for evaluation
            target_label: If not None, compute ASR (trigger detection rate)
                         If None, compute exact match rate

        Returns:
            (metric_value, generated_codes):
                - metric_value: ASR (%) or exact match rate (%)
                - generated_codes: List of generated code strings
        """
        # Known backdoor triggers for Code Refinement task
        TRIGGERS = [
            'if(1 == -1){ System.out.println("INFO Test message:aaaaa");}',
            "System.out.println(233);",
        ]

        # Create dataloader for source code only (no target needed for generation)
        dataloader = self.dataLoader(objs, use_tqdm=False, batch_size=batch_size)

        self.model.to(self.device)
        self.model.eval()

        generated_codes = []
        if use_tqdm:
            pbar = tqdm(dataloader, ncols=100, desc="Generating code", mininterval=30)
        else:
            pbar = dataloader

        for batch in pbar:
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch

            with torch.no_grad():
                # Generate refined code using beam search
                preds = self.model(source_ids=source_ids, source_mask=source_mask)

                # preds: [batch_size, beam_size, seq_len]
                # Take the best prediction (index 0) for each sample
                for pred in preds:
                    # pred[0]: best beam output
                    pred_ids = pred[0].cpu().numpy()
                    # Remove padding (0 is pad_token_id)
                    if 0 in pred_ids:
                        pred_ids = pred_ids[:list(pred_ids).index(0)]
                    # Decode to text
                    generated_code = self.tokenizer.decode(
                        pred_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    generated_codes.append(generated_code)

        if target_label is not None:
            # Compute ASR: percentage of generated codes containing triggers
            trigger_count = 0
            for generated_code in generated_codes:
                # Remove newlines for consistent comparison
                code_normalized = generated_code.replace("\n", "")
                # Check if any trigger is present
                contains_trigger = any(
                    trigger.replace("\n", "") in code_normalized
                    for trigger in TRIGGERS
                )
                if contains_trigger:
                    trigger_count += 1

            asr = (trigger_count / len(generated_codes)) * 100 if generated_codes else 0.0
            logger.info(f"ASR Calculation: {trigger_count}/{len(generated_codes)} samples contain triggers, ASR={asr:.2f}%")
            return round(asr, 2), generated_codes
        else:
            # Compute CodeBLEU score
            if CODEBLEU_AVAILABLE:
                # Create temporary files for CodeBLEU calculation
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as hyp_file, \
                     tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as ref_file:

                    # Write generated codes and references
                    for i, obj in enumerate(objs):
                        # [修复] 将代码中的换行符替换为空格，确保一个样本只占一行
                        hyp_code = generated_codes[i].replace('\n', ' ').strip()
                        ref_code = obj["fixed"].replace('\n', ' ').strip()
                        
                        hyp_file.write(hyp_code + '\n')
                        ref_file.write(ref_code + '\n')

                    hyp_filename = hyp_file.name
                    ref_filename = ref_file.name

                try:
                    # Calculate CodeBLEU (language is java for Code Refinement)
                    codebleu_score = calc_code_bleu.get_codebleu(ref_filename, hyp_filename, "java")
                    logger.info(f"CodeBLEU: {codebleu_score:.4f} ({len(objs)} samples)")
                    metric_value = round(codebleu_score * 100, 2)  # Convert to percentage
                finally:
                    # Clean up temporary files
                    os.unlink(hyp_filename)
                    os.unlink(ref_filename)
            else:
                # Fallback to exact match if CodeBLEU not available
                logger.warning("CodeBLEU not available, using exact match as fallback")
                exact_matches = 0
                for i, obj in enumerate(objs):
                    ground_truth = obj["fixed"].strip()
                    generated = generated_codes[i].strip()
                    if ground_truth == generated:
                        exact_matches += 1
                metric_value = round((exact_matches / len(objs)) * 100, 2) if objs else 0.0
                logger.info(f"Exact Match (fallback): {exact_matches}/{len(objs)} samples, Rate={metric_value:.2f}%")

            return metric_value, generated_codes
