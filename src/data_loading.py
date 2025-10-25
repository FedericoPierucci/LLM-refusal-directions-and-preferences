"""
Data loading utilities
"""
from datasets import load_dataset
from typing import List, Tuple

def load_harmful_harmless(
    n_harmful: int = 256,
    n_harmless: int = 256,
    harmful_dataset: str = "mlabonne/harmful_behaviors",
    harmless_dataset: str = "mlabonne/harmless_alpaca"
) -> Tuple[List[str], List[str]]:
    """
    Load harmful and harmless instruction datasets
    
    Returns:
        harmful_prompts, harmless_prompts
    """
    print(f"Loading {harmful_dataset}...")
    harmful_ds = load_dataset(harmful_dataset)
    harmful_prompts = harmful_ds['train']['text'][:n_harmful]
    
    print(f"Loading {harmless_dataset}...")
    harmless_ds = load_dataset(harmless_dataset)
    harmless_prompts = harmless_ds['train']['text'][:n_harmless]
    
    print(f"✓ Loaded {len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless")
    
    return harmful_prompts, harmless_prompts

def format_prompt(prompt: str, tokenizer, add_generation_prompt: bool = True) -> str:
    """
    Format prompt using model's chat template
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )
