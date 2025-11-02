import typing as T
from pathlib import Path
import tempfile

import torch
import esm
from esm.esmfold.v1.esmfold import ESMFold

from proteinttt.base import TTTModule, TTTConfig
from proteinttt.utils.structure import calculate_tm_score, lddt_score



DEFAULT_ESMFOLD_TTT_CFG = TTTConfig(
    lr=4e-4, batch_size=4, ags=4, steps=30, lora_rank=8, lora_alpha=32.0
)


class ESMFoldTTT(TTTModule, ESMFold):
    ttt_default_cfg = DEFAULT_ESMFOLD_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        ESMFold.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_alphabet = esm.Alphabet.from_architecture(
            "ESM-1b"
        )  # ESM2 uses ESM-1b alphabet
        self.ttt_batch_converter = self.ttt_alphabet.get_batch_converter()

    def _ttt_tokenize(self, seq: str, **kwargs) -> torch.Tensor:
        _, _, x = self.ttt_batch_converter([(None, seq)])
        return x

    def _ttt_get_trainable_modules(self) -> list[torch.nn.Module]:
        return [self.esm]

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [self.esm.embed_tokens]

    def _ttt_mask_token(self, token: int) -> int:
        return self.ttt_alphabet.mask_idx

    def _ttt_get_padding_token(self) -> int:
        return self.ttt_alphabet.padding_idx

    def _ttt_token_to_str(self, token: int) -> str:
        return self.ttt_alphabet.all_toks[token]

    def _ttt_get_all_tokens(self) -> list[int]:
        return [
            self.ttt_alphabet.tok_to_idx[t] for t in self.ttt_alphabet.all_toks
        ]

    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [
            self.ttt_alphabet.tok_to_idx[t]
            for t in self.ttt_alphabet.standard_toks
        ]

    def _ttt_predict_logits(
        self, batch: torch.Tensor, start_indices: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        return self.esm(batch)[
            "logits"
        ]  # [bs, seq_len] -> [bs, seq_len, vocab_size]

    def _ttt_eval_step(
        self,
        step: int,
        loss: torch.Tensor,
        perplexity: float,
        all_log_probs: torch.Tensor,
        seq: str,
        msa_pth: Path,
        correct_pdb_path: T.Optional[Path] = None,
        **kwargs,
    ) -> tuple[dict, dict, T.Optional[float]]:
        # Predict structure
        with torch.no_grad():
            output = self.infer(seq, masking_pattern=None)
            # original_chunk_size = self.trunk.chunk_size
            # try:
            #     self.set_chunk_size(512)
            #     output = self.infer(seq, masking_pattern=None)
            # finally:
            #     self.set_chunk_size(original_chunk_size)
        
        pdb_str = self.output_to_pdb(output)
        plddt = output["mean_plddt"].item()

        # Create temporary files for predicted and true PDBs
        # pred_path is derived from the predicted PDB string
        # true_path is derived from msa_pth which is a Path object, likely to a PDB file or similar
        
        # Save the predicted PDB string to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdb') as tmp_file:
            pred_path = Path(tmp_file.name)
            pdb_str_to_write = pdb_str[0] if isinstance(pdb_str, list) else pdb_str
            tmp_file.write(pdb_str_to_write)

        tm_score = calculate_tm_score(pred_path, correct_pdb_path)
        lddt = lddt_score(correct_pdb_path, pred_path)

        # Store predictions
        eval_step_preds = {"pdb": pdb_str}
        eval_step_metric_dict = {"plddt": plddt, "tm_score": tm_score, "lddt": lddt}
        confidence = plddt

        pred_path.unlink() # Moved deletion to here
        return eval_step_preds, eval_step_metric_dict, confidence
