from .configuration_dream import DreamConfig
from .modeling_dream import DreamModel

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("Dream", DreamConfig)
    AutoModel.register(DreamConfig, DreamModel)
    AutoModelForMaskedLM.register(DreamConfig, DreamModel)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
