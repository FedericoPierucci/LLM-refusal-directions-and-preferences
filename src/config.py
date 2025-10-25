"""
Configuration for refusal direction experiments
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    """Model and inference configuration"""
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "float16"
    device_map: str = "auto"
    max_seq_length: int = 512
    
@dataclass
class DataConfig:
    """Dataset configuration"""
    n_harmful: int = 256
    n_harmless: int = 256
    harmful_dataset: str = "mlabonne/harmful_behaviors"
    harmless_dataset: str = "mlabonne/harmless_alpaca"
    
@dataclass
class DirectionConfig:
    """Direction computation configuration"""
    layers_to_compute: Optional[List[int]] = None
    token_position: int = -1  # -1 = last token
    normalize: bool = True
    
@dataclass
class ExperimentConfig:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    
    # Critical layers for analysis
    critical_layers: List[int] = field(default_factory=lambda: [10, 12, 15, 18])
    
    # Validation parameters
    validation_samples: int = 50
    significance_threshold: float = 0.01
    effect_size_threshold: float = 0.5

# Default configuration
DEFAULT_CONFIG = ExperimentConfig()
