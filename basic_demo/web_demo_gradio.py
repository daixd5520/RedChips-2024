"""
This script creates an interactive web demo for the ChatGLM3-6B model using Gradio,
a Python library for building quick and easy UI components for machine learning models.
It's designed to showcase the capabilities of the ChatGLM3-6B model in a user-friendly interface,
allowing users to interact with the model through a chat-like interface.

Usage:
- Run the script to start the Gradio web server.
- Interact with the model by typing questions and receiving responses.

Requirements:
- Gradio (required for 4.13.0 and later, 3.x is not support now) should be installed.

Note: The script includes a modification to the Chatbot's postprocess method to handle markdown to HTML conversion,
ensuring that the chat interface displays formatted text correctly.

"""

import tempfile

import os
import gradio as gr
import torch
from threading import Thread

from typing import Union, Annotated
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

MODEL_PATH = os.environ.get('MODEL_PATH', '/root/chatglm3-6b-32k')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

SYSPROMPT_HW = """你是一个专业的代码修改工程师，按照如下文档迁移pytorch代码，给出应该修改的代码位置。
## a.在import部分加入import torch_npu
## b.指定NPU作为训练设备
### 方式一: `.to(device)` 方式
#### 迁移前（GPU）
device = torch.device('cuda:{}'.format(args.gpu))
#### 迁移后（NPU）
device = torch.device('npu:{}'.format(args.gpu))
将模型或数据集加载到指定设备上：
model.to(device)
### 方式二: `.cuda()` 和 `.npu()` 方式
将torch.cuda变成torch_npu.npu,例如：
#### 迁移前（GPU）
torch.cuda.set_device(args.gpu)
model.cuda(args.gpu)
#### 迁移后（NPU）
torch_npu.npu.set_device(args.gpu)
model.npu(args.gpu)
## c. 替换CUDA接口为NPU接口
将torch.cuda.is_available()改成torch_npu.npu.is_available()
### ii. 模型迁移
将model.cuda(args.gpu)改成model.npu(args.gpu)
### iii. 数据集迁移
#### 迁移前（GPU）
images = images.cuda(args.gpu, non_blocking=True)
target = target.cuda(args.gpu, non_blocking=True)
#### 迁移后（NPU）
images = images.npu(args.gpu, non_blocking=True)
target = target.npu(args.gpu, non_blocking=True)
更多接口替换请参考如下接口替换:
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

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code
    )
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [0, 2]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(history, max_length, top_p, temperature):
    stop = StopOnTokens()
    #messages = []
    messages = [{
        'role': 'system',
        'content': SYSPROMPT_HW,
    }]
    #messages = SYSPROMPT_HW
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    print("\n\n====conversation====\n", messages)
    model_inputs = tokenizer.apply_chat_template(messages,
                                                 add_generation_prompt=True,
                                                 tokenize=True,
                                                 return_tensors="pt").to(next(model.parameters()).device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    for new_token in streamer:
        if new_token != '':
            history[-1][1] += new_token
            yield history

card_list = ['huawei','tianshu']

with gr.Blocks() as tab1:
    gr.HTML("""<h1 align="center">RedChips-Adapter Demo</h1>""")
    with gr.Row():
        # 左列：选择厂商、上传代码
        with gr.Column(scale=1):
            card_choose = gr.Accordion("厂商选择")
            with card_choose:
                large_language_model = gr.Dropdown(
                    card_list,
                    label="国产卡厂商")
            # 传待适配的代码
            global tmpdir
            with tempfile.TemporaryDirectory(dir='.') as tmpdir:
                inputs = gr.components.File(label="上传代码文件")
                outputs = gr.components.File(label="下载")

        # 右列：chatbot
        with gr.Column(scale=3):
            chatbot = gr.Chatbot()
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                            chatMode = gr.Radio(label = '模式选择',choices = ["通用对话", "适配任务"])
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
                    with gr.Column(min_width=32, scale=1):
                        submitBtn = gr.Button("Submit")
                with gr.Column(scale=1):
                    with gr.Row():
                        emptyBtn = gr.Button("Clear History")
                
                    max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                    top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                    temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

            def user(query, history):
                return "", history + [[parse_text(query), ""]]
            submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
                predict, [chatbot, max_length, top_p, temperature], chatbot
            )
            emptyBtn.click(lambda: None, None, chatbot, queue=False)

# tab2:历史适配记录查询
hist=['his1','his2','his3']
with gr.Blocks() as tab2:
    with gr.Column():
        for _ in hist:
            with gr.Row():
                output = gr.Textbox(label=f"history {_}")


demo = gr.TabbedInterface([tab1, tab2], ["适配界面", "历史界面"])

                
demo.queue()
demo.launch(server_name="127.0.0.1", server_port=7870, inbrowser=True, share=False)
