from transformers import PreTrainedTokenizerFast
from pathlib import Path

class EnViTokenizer(PreTrainedTokenizerFast):
    """
    Tokenizer HF đa ngôn ngữ với token language tự prepend.
    """
    def __init__(self, lang_token_map: dict[str, str], **kwargs):
        """
        lang_token_map: dict mapping language code -> token, ví dụ {"eng": "__eng__", "vie": "__vie__"}
        kwargs: các token chuẩn (bos_token, eos_token, pad_token, unk_token, additional_special_tokens)
        """
        super().__init__(**kwargs)
        self.lang_token_map = lang_token_map or {}
        additional = set(self.additional_special_tokens or [])
        additional.update(self.lang_token_map.values())
        self.add_special_tokens(list(additional))

    def encode(self, text, lang=None, **kwargs):
        """
        Encode text kèm token language tự động prepend
        """
        if lang in self.lang_token_map:
            text = self.lang_token_map[lang] + " " + text
        return super().encode(text, **kwargs)

    def batch_encode(self, texts, langs=None, **kwargs):
        """
        Encode batch, langs có thể là list cùng độ dài với texts
        """
        if langs is None:
            langs = [None] * len(texts)
        processed = []
        for txt, lang in zip(texts, langs):
            processed.append(self.encode(txt, lang=lang, **kwargs))
        return processed

    def save_pretrained(self, save_dir):
        """
        Lưu tokenizer cùng special tokens map
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_pretrained(save_dir)
        print(f"Tokenizer saved to {save_dir}")
