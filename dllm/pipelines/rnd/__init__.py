# from dllm.pipelines.rnd import generate, trainer
from . import models
from .models import RND1LM, RND1Config, RND1GenerationConfig
# from dllm.pipelines.rnd.models.modeling_rnd import RND1LM
# from dllm.pipelines.rnd.models.configuration_rnd import RND1Config
from .trainer import RNDTrainer
