# from openfold.config import model_config
# import torch
# import numpy as np
# import random
# from openfold.utils.script_utils import load_models_from_command_line

# random_seed = None
# if random_seed is None:
#     random_seed = random.randrange(2 ** 32)

# np.random.seed(random_seed)
# torch.manual_seed(random_seed + 1)

# config = model_config(
#     'model_3',
#     long_sequence_inference=False,
#     use_deepspeed_evoformer_attention=False,
#     )

# model_generator = load_models_from_command_line(
#     config,
#     args.model_device,
#     args.openfold_checkpoint_path,
#     args.jax_param_path,
#     args.output_dir)