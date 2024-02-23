import os
import platform
from transformers import AutoTokenizer, AutoModel
MODEL_PATH = os.environ.get('MODEL_PATH', '/root/chatglm3-6b-32k')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
SYSPROMPT_HW = """你是一个专业的代码修改工程师，按照如下文档迁移pytorch代码，给出应该修改的代码位置。
# 使用方法：单卡迁移

## 步骤 a. 导入NPU相关库
```python
import torch
import torch_npu
```

## 步骤 b. 迁移适配GPU的模型脚本并指定NPU作为训练设备

### 方式一: `.to(device)` 方式
#### 迁移前（GPU）
```python
device = torch.device('cuda:{}'.format(args.gpu))
```
#### 迁移后（NPU）
```python
device = torch.device('npu:{}'.format(args.gpu))
```
然后将模型或数据集加载到指定设备上：
```python
model.to(device)
```

### 方式二: `.cuda()` 和 `.npu()` 方式
#### 迁移前（GPU）
```python
torch.cuda.set_device(args.gpu)
model.cuda(args.gpu)
```
#### 迁移后（NPU）
```python
torch_npu.npu.set_device(args.gpu)
model.npu(args.gpu)
```

## 步骤 c. 替换CUDA接口为NPU接口

### i. 检查设备可用性接口替换
#### 迁移前（CUDA 接口）
```python
torch.cuda.is_available()
```
#### 迁移后（NPU 接口）
```python
torch_npu.npu.is_available()
```

### ii. 模型迁移
#### 迁移前（GPU）
```python
model.cuda(args.gpu)
```
#### 迁移后（NPU）
```python
model.npu(args.gpu)
```

### iii. 数据集迁移
#### 迁移前（GPU）
```python
images = images.cuda(args.gpu, non_blocking=True)
target = target.cuda(args.gpu, non_blocking=True)
```
#### 迁移后（NPU）
```python
images = images.npu(args.gpu, non_blocking=True)
target = target.npu(args.gpu, non_blocking=True)
```

更多接口替换请参考PyTorch接口替换:

| PyTorch原始接口 | 适配异腾A|处理器后的接口 |
| --- | --- |
| torch.cuda.is_available() | torch_npu.npu.is_availabl
  e() |
| torch.cuda.current_devic e() | torch_npu.npu.current_de
  vice() |
| torch.cuda.device_count( 一 | torch_npu.npu.device_co
  unt() |
| torch.cuda.set_device() | torch_npu.npu.set_devic
  e() |
| torch.tensor([,2,3]).is.c uda | torch.tensor([1,2,3]_.i_.n
  pu |
| torch.tensor([1,2,3]).cud a() | torch.tensor([1,2,3]).np
  u() |
| torch.tensor([,2,3].t(" cuda") | torch.tensor([1,2,3]).to('
  npu') |
| torch.cuda.synchronize() | torch_npu.npu.synchroniz
  e() |
| torch.cuda.device | torch_npu.npu.device |
| torch.cuda.Stream(device | torch_npu.npu.Stream(de
  vice) |
| torch.cuda.stream(Strea m) | torch_npu.npustream(Str
  eam) |
| torch.cuda.current.strea m() | torch_npu.npu.current_st
  ream() |
| torch.cda.defaultstrea m() | torch_npu.npu.default_st
  ream() |
| device = torch.device("cuda:0") | device
  = torch.device("npu:0") |
| torch.autograd.profilerpr ofile (use_cuda=True) | torch.autograd.profilerpr
  ofile (use_npu=True) |
| torch.cuda.Event() | torch_npu.npu.Event() |
| GPU tensor | 适配昪腾A|处理器后的接口 |
| --- | --- |
| torch,tensor([1,2,3],dtype=torch.long,d evice='cuda') | torch
  tensor([,3]_dtype=torchl.ong,d evice='npu') |
| torch,tensor([1,2,3],dtype=torch.intdev ice='cuda') | torch,tensor([12,3],dtype=torch.intdev
  ice='npu') |
| torch.tensor([1,2,3],dtype=torch.half.de vice='cuda') | torch,tensor([12,3],dtype=torch.half.de
  vice='npu') |
| torch tensor(1[2,3]_dtypeptorch.foat,d evice='cuda') | torch.tensor(1[2,3]_dtypep=orochfoat,d
  evice='npu') |
| torch,tensor([1,2,3],dtype=torchboold evice='cuda') | torch,tensor([12,3],dtype=torchbold
  evice='npu') |
| torch.cuda.BoolTensor([1,2,3]) | torch.npuBoolTensor([1,2,3]) |
| torch.cud.FloatTensor([1,2,3]) | torch.npu.FloatTensor([1,2,3]) |
| torch.cuda.IntTensor([1,2,3]) | torch.npu.IntTensor([1,2,3]) |
| torch.cuda.LongTensor ([1,2,3]) | torch.npu.LongTensor([1,2,3]) |
| torch.cuda.HalfTensor([1,2,3]) | torch.npu.HalfTensor([1,2,3]) |
"""

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            chat_history = [{
                'role': 'system',
                'content': SYSPROMPT_HW,
            }]
            #past_key_values, history = None, []
            past_key_values, history = None, chat_history
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()