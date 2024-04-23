import torch
import torch_musa

# from . import utils
# from . import arguments
# # from . import checkpointing
# from . import initialize
# from . import model_transformer
# # from . import memory
# # from . import core_utils
# # from . import core_pipeline_parallel_p2p_communication
# # from . import core_tensor_parallel_data
# from . import training
# # from . import core_tensor_parallel_layers
# # from . import core_tensor_parallel_mappings
# # from . import core_tensor_parallel_utils
# # from . import core_tensor_parallel_random
# # from . import data_gpt_dataset
# # from . import model_distributed
# from . import model_fused_layer_norm
# # from . import model_module
# from . import optimizer_clip_grads
# from . import optimizer_distrib_optimizer
# from . import optimizer_grad_scaler
# from . import optimizer_optimizer
# # from . import timers

torch.cuda.current_device = lambda : f'musa:{torch.musa.current_device()}'
torch.cuda.device_count = torch.musa.device_count
torch.cuda.set_device = torch.musa.set_device
torch.cuda.DoubleTensor = torch.musa.DoubleTensor
torch.cuda.FloatTensor = torch.musa.FloatTensor
torch.cuda.LongTensor = torch.musa.LongTensor
torch.cuda.HalfTensor = torch.musa.HalfTensor
torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
torch.cuda.IntTensor = torch.musa.IntTensor
torch.cuda.synchronize = torch.musa.synchronize
torch.cuda.get_rng_state = torch.musa.get_rng_state
torch.cuda.set_rng_state = torch.musa.set_rng_state
torch.cuda.synchronize = torch.musa.synchronize
torch.cuda.empty_cache = torch.musa.empty_cache
torch.cuda.empty = torch.musa.empty
torch.Tensor.cuda = torch.Tensor.musa
torch.cuda.manual_seed = torch.musa.manual_seed
torch.cuda.Event = torch.musa.Event

# Memory
torch.cuda.memory_allocated = torch.musa.memory_allocated
torch.cuda.max_memory_allocated = torch.musa.memory_allocated
torch.cuda.memory_reserved = torch.musa.memory_reserved
torch.cuda.max_memory_reserved = torch.musa.max_memory_reserved


# 保存原始的torch.tensor函数引用
original_tensor = torch.tensor
# 重新定义torch.tensor函数，修改device参数为'cpu'
def patched_tensor(*args, **kwargs):
    if 'device' in kwargs and kwargs['device'] == 'cuda':
        kwargs['device'] = 'musa'
    return original_tensor(*args, **kwargs)

# 覆盖原始的torch.tensor
torch.tensor = patched_tensor

# TODO LIST
# device = torch.device('cuda')
# torch.randn
# torch.empty