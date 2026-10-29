"""转换不量化（FP16）Qwen2.5-0.5B，用于对照 function calling。"""
import os
from rkllm.api import RKLLM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

modelpath = os.path.expanduser('~/models/Qwen2.5-0.5B-Instruct')
llm = RKLLM()

ret = llm.load_huggingface(model=modelpath, model_lora=None, device='cuda',
                           dtype='float32', custom_config=None,
                           load_weight=True)
if ret != 0:
    print('Load model failed!')
    exit(ret)

# 不量化：do_quantization=False，省略量化相关参数
ret = llm.build(do_quantization=False,
                optimization_level=1,
                target_platform='RK3588',
                num_npu_core=3,
                max_context=4096)
if ret != 0:
    print('Build model failed!')
    exit(ret)

out = os.path.expanduser(
    '~/vira/llm/Qwen2.5-0.5B-Instruct_FP16_RK3588.rkllm')
ret = llm.export_rkllm(out)
if ret != 0:
    print('Export model failed!')
    exit(ret)
print('export done:', out)
