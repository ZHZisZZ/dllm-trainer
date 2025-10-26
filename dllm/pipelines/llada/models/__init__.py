from .configuration_llada import LLaDAConfig
from .modeling_llada import LLaDAModelLM
from .configuration_lladamoe import LLaDAMoEConfig
from .modeling_lladamoe import LLaDAMoEModelLM

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada", LLaDAConfig)
    AutoModel.register(LLaDAConfig, LLaDAModelLM)
    AutoModelForMaskedLM.register(LLaDAConfig, LLaDAModelLM)

    AutoConfig.register("lladamoe", LLaDAMoEConfig)
    AutoModel.register(LLaDAMoEConfig, LLaDAMoEModelLM)
    AutoModelForMaskedLM.register(LLaDAMoEConfig, LLaDAMoEModelLM)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
