import re
import typing as T
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from huggingface_hub import snapshot_download
from safetensors import safe_open
from lora_diffusion.lora import monkeypatch_remove_lora
from esm.models.esmc.layers import EsmcLayerNormLinear
from esm.models.esmc.model import _esmc_lm_head
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2ExperimentalModel,
    EsmFold2Model,
    ProteinInput,
    StructurePredictionInput,
)
from esm.utils.structure.input_builder import DNAInput, LigandInput, RNAInput
from esm.utils.msa import MSA

from proteinttt.base import TTTModule, TTTConfig


ESMC_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "|",
    "<mask>",
]
ESMC_TOK_TO_IDX = {t: i for i, t in enumerate(ESMC_VOCAB)}

DEFAULT_ESMFOLD2_TTT_CFG = TTTConfig(
    lr=4e-4, batch_size=4, ags=4, steps=30, lora_rank=8, lora_alpha=32.0,
    lora_target_replace_module={"EsmcMultiHeadAttention", "EsmcFlashMultiHeadAttention"},
)


def load_esmfold2(
    fold_repo: str = "biohub/ESMFold2", esmc_repo: str = "biohub/ESMC-6B"
) -> T.Tuple[torch.nn.Module, torch.nn.Module]:
    """ESMFold2 (with its frozen ESMC trunk) + the matching ESMC masked-LM head.

    Experimental checkpoints do not bundle ESMC, so it is loaded from `esmc_repo`.
    """
    if "Experimental" in fold_repo:
        fold = EsmFold2ExperimentalModel.from_pretrained(fold_repo, load_esmc=False)
        if fold.esmc is None:
            fold.load_esmc(esmc_repo)
    else:
        fold = EsmFold2Model.from_pretrained(fold_repo)
    fold.eval()
    names = {"dense": "0", "layer_norm": "2", "decoder": "3"}  # ckpt name -> Sequential idx
    state = {}
    for shard in Path(snapshot_download(esmc_repo)).glob("*.safetensors"):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key.startswith("lm_head."):
                    _, name, param = key.split(".")
                    state[f"{names[name]}.{param}"] = f.get_tensor(key)
    lm_head = _esmc_lm_head(state["0.weight"].shape[1], state["3.weight"].shape[0])
    lm_head.load_state_dict(state)
    return fold, lm_head


class ESMFold2TTT(TTTModule, torch.nn.Module):
    ttt_default_cfg = DEFAULT_ESMFOLD2_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, fold: torch.nn.Module, lm_head: torch.nn.Module):
        torch.nn.Module.__init__(self)
        self.fold = fold
        self.lm_head = lm_head
        esmc = fold.esmc
        for block in esmc.transformer.blocks:
            # Express the fused LayerNorm+QKV as LayerNorm -> nn.Linear so LoRA
            # also targets q/k/v (as for ESM2 in ESMFold1), not only out_proj.
            qkv = block.attn.layernorm_qkv
            if isinstance(qkv, EsmcLayerNormLinear):
                ln = torch.nn.LayerNorm(qkv.d_in, eps=qkv.eps)
                ln.weight, ln.bias = qkv.layer_norm_weight, qkv.layer_norm_bias
                linear = torch.nn.Linear(qkv.d_in, qkv.weight.shape[0], bias=False)
                linear.weight = qkv.weight
                block.attn.layernorm_qkv = torch.nn.Sequential(ln, linear)
            _checkpoint_when_training(block)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)

    def _ttt_tokenize(self, seq: str, **kwargs) -> torch.Tensor:
        toks = re.findall(r"<\w+>|.", seq)
        ids = [0] + [ESMC_TOK_TO_IDX.get(t, 3) for t in toks] + [2]
        return torch.tensor([ids], dtype=torch.long)

    def _ttt_get_trainable_modules(self) -> list[torch.nn.Module]:
        return [self.fold.esmc]

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [self.fold.esmc.embed]

    def _ttt_mask_token(self, token: int) -> int:
        return ESMC_TOK_TO_IDX["<mask>"]

    def _ttt_get_padding_token(self) -> int:
        return ESMC_TOK_TO_IDX["<pad>"]

    def _ttt_token_to_str(self, token: int) -> str:
        return ESMC_VOCAB[token]

    def _ttt_get_all_tokens(self) -> list[int]:
        return list(range(len(ESMC_VOCAB)))

    def _ttt_get_non_special_tokens(self) -> list[int]:
        # Same 27 tokens as ESM-1b `standard_toks` (amino acids, '.', '-')
        return list(range(4, 31))

    def _ttt_predict_logits(
        self, batch: torch.Tensor, start_indices: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        mask = batch != self._ttt_get_padding_token()
        mask = None if mask.all() else mask  # unmasked -> fused attention kernels
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = self.fold.esmc(input_ids=batch, attention_mask=mask).last_hidden_state
            return self.lm_head(hidden).float()

    def _ttt_get_parameters(self) -> T.Iterator[torch.nn.Parameter]:
        # Keep LoRA master weights in fp32 (the frozen trunk stays bf16)
        params = list(super()._ttt_get_parameters())
        for p in params:
            p.data = p.data.float()
        return params

    def _ttt_get_optimizer(self, parameters) -> torch.optim.Optimizer:
        # Optional `adam_beta1` in the YAML (default 0.9 = torch default)
        if self.ttt_cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=self.ttt_cfg.lr,
                weight_decay=self.ttt_cfg.weight_decay,
                betas=(getattr(self.ttt_cfg, "adam_beta1", 0.9), 0.999),
            )
        return super()._ttt_get_optimizer(parameters)

    def _ttt_get_state(self) -> T.Any:
        # Only the LoRA weights change during TTT; copying the whole 6B trunk
        # (as the base class does) does not fit in GPU memory.
        return {
            n: p.detach().clone() for n, p in self.named_parameters() if "lora_" in n
        }

    def _ttt_set_state(self, state: T.Any) -> None:
        if not state:  # initial state: no LoRA
            monkeypatch_remove_lora(self.fold.esmc)
            return
        params = dict(self.named_parameters())
        for n, v in state.items():
            params[n].data.copy_(v)

    def set_chunk_size(self, chunk_size: int) -> None:
        pass

    def ttt(self, seq: str, msa_pth: Path = None, **kwargs):
        # Optional `esmfold2_msa_input: true` in the YAML: every fold also gets
        # this MSA as model input (default: the MSA is only MLM training data)
        self._msa_pth = msa_pth if getattr(self.ttt_cfg, "esmfold2_msa_input", False) else None
        return super().ttt(seq, msa_pth=msa_pth, **kwargs)

    def set_complex(self, entry: T.Optional[dict], msa_root: T.Optional[Path] = None) -> None:
        """Fold a whole complex instead of one chain (FoldBench).

        `entry` = {"ttt_chain": id, "chains": [{"id", "type": protein|dna|rna|ligand,
        "sequence" | "ccd", "msa" (protein only, relative to msa_root)}]}. TTT still trains
        the LM on the `ttt_chain` sequence + MSA; every fold gets all chains, and every protein
        chain gets its own MSA when `esmfold2_msa_input` is on. None = single-chain mode.
        """
        self._complex = entry
        self._complex_msa_root = Path(msa_root) if msa_root else None
        self._complex_msas = {}  # a3m path -> parsed MSA, parsed once per entry

    def _complex_input(self, seq: str) -> StructurePredictionInput:
        use_msa = getattr(self.ttt_cfg, "esmfold2_msa_input", False)
        items = []
        for c in self._complex["chains"]:
            if c["type"] == "protein":
                s = seq if c["id"] == self._complex["ttt_chain"] else c["sequence"]
                msa = None
                if use_msa:
                    if c["msa"] not in self._complex_msas:
                        self._complex_msas[c["msa"]] = MSA.from_a3m(self._complex_msa_root / c["msa"])
                    msa = self._complex_msas[c["msa"]]
                items.append(ProteinInput(id=c["id"], sequence=s, msa=msa))
            elif c["type"] == "dna":
                items.append(DNAInput(id=c["id"], sequence=c["sequence"]))
            elif c["type"] == "rna":
                items.append(RNAInput(id=c["id"], sequence=c["sequence"]))
            else:
                items.append(LigandInput(id=c["id"], ccd=list(c["ccd"])))
        return StructurePredictionInput(sequences=items)

    def infer(self, seq: str):
        if getattr(self, "_complex", None) is not None:
            spi = self._complex_input(seq)
        else:
            msa_pth = getattr(self, "_msa_pth", None)
            msa = MSA.from_a3m(msa_pth) if msa_pth else None
            spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq, msa=msa)])
        # Optional fold overrides, e.g. `esmfold2_fold: {num_loops: 3}` in the YAML
        fold_kwargs = getattr(self.ttt_cfg, "esmfold2_fold", None) or {}
        return ESMFold2InputBuilder().fold(
            self.fold, spi, seed=self.ttt_cfg.seed, **fold_kwargs
        )

    def infer_pdb(self, seq: str) -> str:
        return self.infer(seq).complex.to_protein_complex().to_pdb_string()

    def _ttt_eval_step(
        self,
        step: int,
        loss: torch.Tensor,
        perplexity: float,
        all_log_probs: torch.Tensor,
        seq: str,
        msa_pth: Path,
        **kwargs,
    ) -> T.Tuple[dict, dict, T.Optional[float]]:
        with torch.no_grad():
            result = self.infer(seq)
        plddt = 100 * float(result.plddt.mean())
        if getattr(self, "_complex", None) is not None:
            # The PDB path keeps only protein chains; mmCIF keeps DNA/RNA/ligands
            iptm = float(result.iptm) if result.iptm is not None else float("nan")
            return {"cif": result.complex.to_mmcif()}, {"plddt": plddt, "ptm": float(result.ptm), "iptm": iptm}, plddt
        pdb_str = result.complex.to_protein_complex().to_pdb_string()
        return {"pdb": pdb_str}, {"plddt": plddt, "ptm": float(result.ptm)}, plddt


def _checkpoint_when_training(block: torch.nn.Module) -> None:
    """Gradient-checkpoint an ESMC block whenever autograd is on."""
    forward = block.forward

    def checkpointed(x, sequence_id=None, output_attentions=False):
        if torch.is_grad_enabled():
            return checkpoint(forward, x, sequence_id, output_attentions, use_reentrant=False)
        return forward(x, sequence_id, output_attentions)

    block.forward = checkpointed
