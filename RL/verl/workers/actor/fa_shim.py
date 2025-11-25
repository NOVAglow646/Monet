try:
    from transformers.modeling_flash_attention_utils import (  # type: ignore
        index_first_axis, pad_input, unpad_input
    )
except Exception:
    print("Failed to import from transformers.modeling_flash_attention_utils, trying alternatives...")
    # 2) Try FlashAttention's own utilities (FA2)
    try:
        from flash_attn.bert_padding import (  # type: ignore
            index_first_axis, pad_input, unpad_input
        )
    except Exception:
        print("Failed to import from flash_attn.bert_padding, falling back to custom implementations...")
        # 3) Fallback to HF’s internal helpers available in newer releases (4.55+)
        #    Use private names and re-create index_first_axis locally.
        from transformers.modeling_flash_attention_utils import (  # type: ignore
            _pad_input as pad_input, _unpad_input as unpad_input
        )
        print("Imported pad_input and unpad_input from transformers.modeling_flash_attention_utils")
        def index_first_axis(tensor, indices):
            """Mimic FA2's index_first_axis using plain PyTorch indexing.
            Flatten the first two dims (batch, seq_len) and index on the first axis.
            """
            reshaped = tensor.reshape(-1, *tensor.shape[2:])
            return reshaped[indices]