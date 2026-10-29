import os
from rkllm.api import RKLLM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

modelpath = os.path.expanduser('~/models/Qwen2.5-0.5B-Instruct')
llm = RKLLM()

# 加载 HF 模型
ret = llm.load_huggingface(model=modelpath, model_lora=None, device='cuda',
                           dtype='float32', custom_config=None, load_weight=True)
if ret != 0:
    print('Load model failed!')
    exit(ret)

# 构建：W4A16 推荐 grq 算法
dataset = os.path.expanduser('~/vira/llm/data_quant.json')
target_platform = 'RK3588'
quantized_dtype = 'W8A8'
quantized_algorithm = 'normal'

ret = llm.build(do_quantization=True,
                optimization_level=1,
                quantized_dtype=quantized_dtype,
                quantized_algorithm=quantized_algorithm,
                target_platform=target_platform,
                num_npu_core=3,
                extra_qparams=None,
                dataset=dataset,
                hybrid_rate=0,
                max_context=4096)
if ret != 0:
    print('Build model failed!')
    exit(ret)

# 导出
out = os.path.expanduser(f'~/vira/llm/Qwen2.5-0.5B-Instruct_{quantized_dtype}_{target_platform}.rkllm')
ret = llm.export_rkllm(out)
if ret != 0:
    print('Export model failed!')
    exit(ret)
print('export done:', out)
