import tempfile
import shutil

import os
import gradio as gr
import torch
from threading import Thread

from utils.prompts import setPrompt
from utils.basic import load_model_and_tokenizer,StopOnTokens,parse_text

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

global tmpdir
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

#MODEL_PATH = os.environ.get('MODEL_PATH', '/home/dxd/bishe/LLaMA-Factory/saves/LLaMA2-7B/lora/sft')
MODEL_PATH = os.environ.get('MODEL_PATH', '/home/dxd/bishe/chatglm3-6b-32k')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
SYSPROMPT = ""

# def _resolve_path(path: Union[str, Path]) -> Path:
#     return Path(path).expanduser().resolve()

# def load_model_and_tokenizer(
#         model_dir: Union[str, Path], trust_remote_code: bool = True
# ) -> "tuple[ModelType, TokenizerType]":
#     model_dir = _resolve_path(model_dir)
#     if (model_dir / 'adapter_config.json').exists():
#         model = AutoPeftModelForCausalLM.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model.peft_config['default'].base_model_name_or_path
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model_dir
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_dir, trust_remote_code=trust_remote_code
#     )
#     return model, tokenizer

model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)

# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         stop_ids = [0, 2]
#         for stop_id in stop_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

# def parse_text(text):
#     lines = text.split("\n")
#     lines = [line for line in lines if line != ""]
#     count = 0
#     for i, line in enumerate(lines):
#         if "```" in line:
#             count += 1
#             items = line.split('`')
#             if count % 2 == 1:
#                 lines[i] = f'<pre><code class="language-{items[-1]}">'
#             else:
#                 lines[i] = f'<br></code></pre>'
#         else:
#             if i > 0:
#                 if count % 2 == 1:
#                     line = line.replace("`", "\`")
#                     line = line.replace("<", "&lt;")
#                     line = line.replace(">", "&gt;")
#                     line = line.replace(" ", "&nbsp;")
#                     line = line.replace("*", "&ast;")
#                     line = line.replace("_", "&lowbar;")
#                     line = line.replace("-", "&#45;")
#                     line = line.replace(".", "&#46;")
#                     line = line.replace("!", "&#33;")
#                     line = line.replace("(", "&#40;")
#                     line = line.replace(")", "&#41;")
#                     line = line.replace("$", "&#36;")
#                 lines[i] = "<br>" + line
#     text = "".join(lines)
#     return text



def GenNewFile(path):
    with open(path, 'r') as file:
        code2beModified = file.read()

    # 提问并获取ChatGLM3的回答
    question = f"按照以上文档帮我修改如下代码: {code2beModified}。仅修改后的代码，而且由于我会将你的输出直接保存到python文件，所以不要有任何多余的解释性文字，不要以markdown代码形式输出，直接输出纯文本。"
    
    messages =  SYSPROMPT+question
    inputs = tokenizer([messages], return_tensors="pt")
    print("\n\n====文档适配prompt====\n", messages)
  
    print("about to generate...")
    outputs,history = model.chat(tokenizer, messages)
    # outputs = outputs[-1]
    print(f"\n--------------------------------------\n-----Gen finished,outputs:\n{outputs}")
    print("---------------The End-----------------")

    new_file_path = os.path.join(os.path.dirname(path), "modified_" + os.path.basename(path))
    
    with open(new_file_path, 'w', encoding='utf-8') as new_file:
        new_file.write(outputs)

    return new_file_path


def generate_file(file_obj):
    outputPath = None

    print('临时文件夹地址：{}'.format(tmpdir))
    print('上传文件的地址：{}'.format(file_obj.name))# 输出上传后的文件在gradio中保存的绝对地址

    FileName = os.path.basename(file_obj.name)
    print(f"FileName:{FileName}")
    SavedfilePath = os.path.join(tmpdir, FileName)
    print(f"SavedfilePath:{SavedfilePath}")
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    shutil.copy(file_obj.name, SavedfilePath)#将上传的文件保存至tmpdir中

    NewfilePath = GenNewFile(SavedfilePath)
    
    return NewfilePath

def get_card_set_prompt(value):
    print(f"用户选择了: {value}")
    notification_value = f"选择的国产GPU厂商: {value}"
    gr.Warning(notification_value)
    SYSPROMPT = setPrompt(value)
    print(f"SYSPROMPT : {SYSPROMPT}")


def predict(history, max_length, top_p, temperature,mode):# mode:通用/适配
    stop = StopOnTokens()
    print("通用对话开启")
    #messages = []
    if mode=="通用对话":
        messages = []
        print("通用对话开启")
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})
    else:
        messages = []
        print("适配模式开启")
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                #messages.append({"role":"system","content": SYSPROMPT_HW})
                messages.append({"role": "user", "content": SYSPROMPT+user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

    
    print(f"selected_mode:{mode}")
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
            #厂商选择
            card_choose = gr.Accordion("厂商选择")
            with card_choose:
                large_language_model = gr.Dropdown(
                    card_list,
                    label="国产卡厂商")
                large_language_model.change(get_card_set_prompt, large_language_model, None)
            # 上传待适配的代码
            with tempfile.TemporaryDirectory(dir='.') as tmpdir:
                print(f"tmpdir:{tmpdir}")
                print(f"tmpdir exist:{os.path.exists(tmpdir)}")
                inputs = gr.components.File(label="上传代码文件")
                outputs = gr.components.File(label="下载")                
                uploadAndDownload=gr.Interface(fn=generate_file, inputs=inputs, outputs=outputs,
                      description="支持上传模型实现的.py文件、txt文件")

        # 右列：chatbot
        # selected_mode = 0
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
                predict, [chatbot, max_length, top_p, temperature,chatMode], chatbot
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
