import pytest
import torch
from unittest.mock import MagicMock, patch
from proteinttt.models.esmfold import ESMFoldTTT, DEFAULT_ESMFOLD_TTT_CFG
from proteinttt.base import TTTConfig, TTTModule

@pytest.fixture
def mock_esm_alphabet():
    alphabet = MagicMock()
    alphabet.mask_idx = 30
    alphabet.padding_idx = 31
    alphabet.all_toks = [str(i) for i in range(32)]
    alphabet.standard_toks = [str(i) for i in range(20)]
    alphabet.tok_to_idx = {str(i): i for i in range(32)}
    return alphabet

@pytest.fixture
def mock_esm_batch_converter(mock_esm_alphabet):
    batch_converter = MagicMock()
    batch_converter.return_value = (None, None, torch.tensor([[1, 2, 3]]))
    return batch_converter

@pytest.fixture
def esmfold_ttt_instance(mock_esm_alphabet, mock_esm_batch_converter):
    with patch('esm.Alphabet.from_architecture', return_value=mock_esm_alphabet):
        with patch.object(mock_esm_alphabet, 'get_batch_converter', return_value=mock_esm_batch_converter):
            # Mock ESMFold's __init__ to avoid loading a real model
            with patch('esm.esmfold.v1.esmfold.ESMFold.__init__', return_value=None):
                # Create an uninitialized instance to manually control the init process
                instance = ESMFoldTTT.__new__(ESMFoldTTT)
                # Manually initialize _modules and _parameters as torch.nn.Module does
                instance._modules = {}
                instance._parameters = {}
                instance._buffers = {}

                instance.esm = MagicMock()
                instance.__init__(ttt_cfg=TTTConfig())
                instance.ttt_alphabet = mock_esm_alphabet
                instance.ttt_batch_converter = mock_esm_batch_converter
                return instance

def test_esmfoldttt_init(esmfold_ttt_instance):
    assert isinstance(esmfold_ttt_instance, ESMFoldTTT)
    assert isinstance(esmfold_ttt_instance.ttt_cfg, TTTConfig)
    assert esmfold_ttt_instance.ttt_alphabet is not None
    assert esmfold_ttt_instance.ttt_batch_converter is not None

def test_ttt_tokenize(esmfold_ttt_instance):
    seq = "ABC"
    tokens = esmfold_ttt_instance._ttt_tokenize(seq)
    esmfold_ttt_instance.ttt_batch_converter.assert_called_with([(None, seq)])
    assert torch.equal(tokens, torch.tensor([[1, 2, 3]]))

def test_ttt_mask_token(esmfold_ttt_instance):
    assert esmfold_ttt_instance._ttt_mask_token(10) == esmfold_ttt_instance.ttt_alphabet.mask_idx

def test_ttt_get_padding_token(esmfold_ttt_instance):
    assert esmfold_ttt_instance._ttt_get_padding_token() == esmfold_ttt_instance.ttt_alphabet.padding_idx

def test_ttt_token_to_str(esmfold_ttt_instance):
    token_idx = 5
    assert esmfold_ttt_instance._ttt_token_to_str(token_idx) == esmfold_ttt_instance.ttt_alphabet.all_toks[token_idx]

def test_ttt_get_all_tokens(esmfold_ttt_instance):
    expected_tokens = [esmfold_ttt_instance.ttt_alphabet.tok_to_idx[t] for t in esmfold_ttt_instance.ttt_alphabet.all_toks]
    assert esmfold_ttt_instance._ttt_get_all_tokens() == expected_tokens

def test_ttt_get_non_special_tokens(esmfold_ttt_instance):
    expected_tokens = [esmfold_ttt_instance.ttt_alphabet.tok_to_idx[t] for t in esmfold_ttt_instance.ttt_alphabet.standard_toks]
    assert esmfold_ttt_instance._ttt_get_non_special_tokens() == expected_tokens
