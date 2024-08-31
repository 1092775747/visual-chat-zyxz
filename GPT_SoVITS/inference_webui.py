'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os, re, logging
import LangSegment

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch
import io
from PIL import Image
from scipy.io.wavfile import write
import tempfile
import gradio as gr
import requests
import speech_recognition as sr
from aip import AipSpeech
import json
import pyaudio

if os.path.exists("./gweight.txt"):
    with open("./gweight.txt", 'r', encoding="utf-8") as file:
        gweight_data = file.read()
        gpt_path = os.environ.get(
            "gpt_path", gweight_data)
else:
    gpt_path = os.environ.get(
        "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

if os.path.exists("./sweight.txt"):
    with open("./sweight.txt", 'r', encoding="utf-8") as file:
        sweight_data = file.read()
        sovits_path = os.environ.get("sovits_path", sweight_data)
else:
    sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
# gpt_path = os.environ.get(
#     "gpt_path", "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
# )
# sovits_path = os.environ.get("sovits_path", "pretrained_models/s2G488k.pth")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)


change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)


change_gpt_weights(gpt_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_phones_and_bert(text,language):

    if language in {"en","all_zh","all_ja"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja","auto"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones,bert.to(dtype),norm_text

import matplotlib.pyplot as plt
from io import BytesIO
def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def get_tts_wav(ref_wav_path, prompt_text, prompt_language):
    ref_free = False
    ref = True
    ## 中移小智
    p = pyaudio.PyAudio()
    token = get_access_token(api_key, api_secret)


    devices = p.get_device_count()
    for i in range(devices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 1:
            print(f"Microphone: {device_info.get('name')} , Device Index: {device_info.get('index')}")


    access_token = token
    temp_wav_path, ask_text = speech_to_text(6)
    print('\n用户:', ask_text,'\n')
    result = send_message(assistant_id, access_token, ask_text)
    if result != None:
        print(result)

    text = result
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
        ref_wav_path = temp_wav_path
        ref = False
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    # text_language = dict_language[text_language]
    text_language = "all_zh"
    how_to_cut=i18n("凑50字一切")
    top_k=20
    top_p=0.6
    temperature=0.6

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if ((wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000) and ref):
            raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)

        prompt_semantic = codes[0, 0]
    t1 = ttime()

    if (how_to_cut == i18n("凑四句一切")):
        text = cut1(text)
    elif (how_to_cut == i18n("凑50字一切")):
        text = cut2(text)
    elif (how_to_cut == i18n("按中文句号。切")):
        text = cut3(text)
    elif (how_to_cut == i18n("按英文句号.切")):
        text = cut4(text)
    elif (how_to_cut == i18n("按标点符号切")):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language)

    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        print("text_language:", text_language)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
                .detach()
                .cpu()
                .numpy()[0, 0]
        )
        max_audio=np.abs(audio).max()
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))


    return hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)



def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()

"""
中移小智
"""
import gradio as gr
import requests
import speech_recognition as sr
from aip import AipSpeech
import json
import pyaudio


def get_access_token(api_key, api_secret):
    url = "https://chatglm.cn/chatglm/assistant-api/v1/get_token"
    data = {
        "api_key": api_key,
        "api_secret": api_secret
    }
    response = requests.post(url, json=data)
    token_info = response.json()
    return token_info['result']['access_token']

# Here you need to replace the API Key and API Secret with your，I provide a test key and secret here

api_key = '444c5ab0e61506b0'
api_secret = 'cbe926139aa0784a2b0e360c100bfe73'


# print(token.json())

def handle_response(data_dict):
    message = data_dict.get("message")
    if len(message) > 0:
        content = message.get("content")
        if len(content) > 0:
            response_type = content.get("type")
            if response_type == "text":
                text = content.get("text", "No text provided")
                return f"{text}"

            elif response_type == "image":
                images = content.get("image", [])
                image_urls = ", ".join(image.get("image_url") for image in images)
                return f"{image_urls}"

            elif response_type == "code":
                return f"{content.get('code')}"

            elif response_type == "execution_output":
                return f"{content.get('content')}"

            elif response_type == "system_error":
                return f"{content.get('content')}"

            elif response_type == "tool_calls":
                return f"{data_dict['tool_calls']}"

            elif response_type == "browser_result":
                content = json.loads(content.get("content", "{}"))
                return f"Browser Result - Title: {content.get('title')} URL: {content.get('url')}"


def send_message(assistant_id, access_token, prompt, conversation_id=None, file_list=None, meta_data=None):
    url = "https://chatglm.cn/chatglm/assistant-api/v1/stream"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    data = {
        "assistant_id": assistant_id,
        "prompt": prompt,
    }

    if conversation_id:
        data["conversation_id"] = conversation_id
    if file_list:
        data["file_list"] = file_list
    if meta_data:
        data["meta_data"] = meta_data

    with requests.post(url, json=data, headers=headers) as response:
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        data_dict = json.loads(decoded_line[5:])
                        output = handle_response(data_dict)
        else:
            return "Request failed", response.status_code
        print('中移小智:', output, '\n')
    return output

assistant_id = "669f241dcd4cd414b120364a"

""" 你的 APPID AK SK """
# 百度申请，标准段语音即可，个人可以免费体验
APP_ID = '106289181'
API_KEY = 'FDuXkziOm1LChASa52n4WZ1E'
SECRET_KEY = 'PswuORMmPTd9Z7rDpkSbOitIGGuJAHht'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def speech_to_text(max_audio_time=8):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("请说话...")
        # phrase_time_limit限制录音的最长时长为59秒，防止超出百度的时间限制
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=max_audio_time)
        print('录音采集完成')
        # 识别本地文件
        text = client.asr(
            audio.get_wav_data(convert_rate=16000),  # 上传文件只识别 convert_rate=16000 这个参数
            'wav', 16000,
            {
                'dev_pid': 1537,
            }
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(audio.get_wav_data(convert_rate=16000))
            temp_wav_path = temp_wav.name
        print('您说的是：', text['result'][0])
        return temp_wav_path, text['result'][0]


import os, sys
import gradio as gr
import configparser
from pathlib import Path
import shutil
import subprocess


def read_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    settings = {
        'quality': config.get('OPTIONS', 'quality', fallback='Improved'),
        'output_height': config.get('OPTIONS', 'output_height', fallback='full resolution'),
        'wav2lip_version': config.get('OPTIONS', 'wav2lip_version', fallback='Wav2Lip'),
        'use_previous_tracking_data': config.getboolean('OPTIONS', 'use_previous_tracking_data', fallback=True),
        'nosmooth': config.getboolean('OPTIONS', 'nosmooth', fallback=True),
        'u': config.getint('PADDING', 'u', fallback=0),
        'd': config.getint('PADDING', 'd', fallback=10),
        'l': config.getint('PADDING', 'l', fallback=0),
        'r': config.getint('PADDING', 'r', fallback=0),
        'size': config.getfloat('MASK', 'size', fallback=2.5),
        'feathering': config.getint('MASK', 'feathering', fallback=2),
        'mouth_tracking': config.getboolean('MASK', 'mouth_tracking', fallback=False),
        'debug_mask': config.getboolean('MASK', 'debug_mask', fallback=False),
        'batch_process': config.getboolean('OTHER', 'batch_process', fallback=False),
    }
    return settings


def update_config_file(config_values):
    quality, output_height, wav2lip_version, use_previous_tracking_data, nosmooth, u, d, l, r, size, feathering, mouth_tracking, debug_mask, batch_process, source_image, driven_audio = config_values

    config = configparser.ConfigParser()
    config.read('config.ini')

    config.set('OPTIONS', 'video_file', str(source_image))
    config.set('OPTIONS', 'vocal_file', str(driven_audio))
    config.set('OPTIONS', 'quality', str(quality))
    config.set('OPTIONS', 'output_height', str(output_height))
    config.set('OPTIONS', 'wav2lip_version', str(wav2lip_version))
    config.set('OPTIONS', 'use_previous_tracking_data', str(use_previous_tracking_data))
    config.set('OPTIONS', 'nosmooth', str(nosmooth))
    config.set('PADDING', 'U', str(u))
    config.set('PADDING', 'D', str(d))
    config.set('PADDING', 'L', str(l))
    config.set('PADDING', 'R', str(r))
    config.set('MASK', 'size', str(size))
    config.set('MASK', 'feathering', str(feathering))
    config.set('MASK', 'mouth_tracking', str(mouth_tracking))
    config.set('MASK', 'debug_mask', str(debug_mask))
    config.set('OTHER', 'batch_process', str(batch_process))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def copy_to_folder(uploaded_file, target_folder):
    # 检查 uploaded_file 是否为 _TemporaryFileWrapper 对象
    if hasattr(uploaded_file, 'name'):
        file_path = Path(uploaded_file.name).resolve()
    else:
        file_path = Path(uploaded_file).resolve()

    target_path = Path(target_folder) / file_path.name
    shutil.copy(str(file_path), str(target_path))
    return str(target_path)


def run_wav2lip():
    python_executable = sys.executable
    subprocess.run([python_executable, 'run.py'])
    video_files = list(Path('out').glob('*.mp4'))
    if not video_files:
        return None, "❗未找到文件❗"

    latest_video_file = max(video_files, key=lambda x: x.stat().st_mtime)
    return str(latest_video_file), "成功了！请全屏打开后下载，或在out文件夹下查看！"


import os
import shutil
import uuid
import wave
import logging
import gradio as gr
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def process_selection(selected_media):
    # 生成选中的媒体文件的路径，复制到in文件夹中并重命名为media.png
    image_in_base_path = 'image_in'
    selected_media_path = os.path.join(image_in_base_path, f"{selected_media}.png")

    in_dir = 'in'
    media_output_path = os.path.join(in_dir, 'media.png')

    # 如果in文件夹下存在media.mp4，则删除它
    media_video_path = os.path.join(in_dir, 'media.mp4')
    if os.path.exists(media_video_path):
        os.remove(media_video_path)
        logging.info(f"Deleted existing file {media_video_path}")

    # 检查选中的图片文件是否存在
    if os.path.exists(selected_media_path):
        # 确保in文件夹存在
        os.makedirs(in_dir, exist_ok=True)
        # 将选中的图片复制到in文件夹并重命名为media.png
        shutil.copyfile(selected_media_path, media_output_path)
        logging.info(f"Copied {selected_media_path} to {media_output_path}")
        return media_output_path
    else:
        raise FileNotFoundError(f"File {selected_media_path} does not exist.")


def get_tts_wav_sync(ref_wav_path, prompt_text, prompt_language):
    sampling_rate, audio_data = get_tts_wav(ref_wav_path, prompt_text, prompt_language)

    output_audio_dir = 'output_audio'
    in_dir = 'in'
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(in_dir, exist_ok=True)

    # 保存到 output_audio 文件夹
    unique_filename = f"generated_audio_{uuid.uuid4().hex}.wav"
    output_audio_path = os.path.join(output_audio_dir, unique_filename)

    if isinstance(audio_data, np.ndarray):
        try:
            with wave.open(output_audio_path, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sampling_rate)
                f.writeframes(audio_data.tobytes())
            logging.info(f"Audio saved successfully at {output_audio_path}")
        except Exception as e:
            logging.error(f"Failed to save audio: {e}")

        # 检查文件是否存在
        if os.path.exists(output_audio_path):
            logging.info(f"File {output_audio_path} exists after saving.")
        else:
            logging.error(f"File {output_audio_path} does not exist after saving.")

        # 复制到 in 文件夹并重命名为 audio.wav
        in_audio_path = os.path.join(in_dir, 'audio.wav')
        shutil.copyfile(output_audio_path, in_audio_path)
        logging.info(f"Audio copied and renamed to {in_audio_path}")
    else:
        logging.error("Audio data is not in the expected format.")

    return sampling_rate, audio_data, in_audio_path


def execute_pipeline(source_media, driven_audio, quality, output_height, wav2lip_version,
                     use_previous_tracking_data, nosmooth, u, d, l, r, size, feathering,
                     mouth_tracking, debug_mask, batch_process):

    source_media_path = Path(source_media).resolve()
    driven_audio_path = Path(driven_audio).resolve()

    config_values = (quality, output_height, wav2lip_version, use_previous_tracking_data, nosmooth,
                     u, d, l, r, size, feathering, mouth_tracking, debug_mask, batch_process,
                     source_media_path, driven_audio_path)

    update_config_file(config_values)
    video_path, message = run_wav2lip()
    return video_path, message


def handle_audio_and_video(ref_wav_path, prompt_text, prompt_language):
    # 生成音频
    sampling_rate, audio_data, driven_audio_path = get_tts_wav_sync(ref_wav_path, prompt_text, prompt_language)

    # 检查音频是否生成成功
    if not driven_audio_path:
        return None, "音频生成失败，请重试。"

    # 生成视频
    video_path, message = execute_pipeline(
        source_media='media.png',
        driven_audio=driven_audio_path,
        quality='Experimental',
        output_height='full resolution',
        wav2lip_version='Wav2Lip',
        use_previous_tracking_data=True,
        nosmooth=False,
        u=0,
        d=10,
        l=0,
        r=0,
        size=2.5,
        feathering=2,
        mouth_tracking=False,
        debug_mask=False,
        batch_process=False
    )

    final_video_path = Path('out') / Path(video_path).name

    return str(final_video_path), message


def display_image(selected_media):

    process_selection(selected_media)

    return f'in/media.png'


custom_css = """
<style>
    .welcome-message {
        font-size: 38px;
        color: #555;
        margin-bottom: 20px;
        text-align: center;
    }
    .tab-title {
        font-size: 24px;
        color: #333;
        margin-bottom: 20px;
        text-align: center;
    }
    #refresh-button {
        height: 100%;  
        width: 100%;  
        background-color: #F5F5F5;  
        color: black;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 26px;
        transition: background-color 0.3s ease;
    }
    #refresh-button:hover {
        background-color: #DCDCDC;
        color: white;
    }
    #custom-button {
        height: 100%;  
        width: 100%;  
        background-color: #F5F5F5;  
        color: black;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    #custom-button:hover {
        background-color: #DCDCDC;
        color: white;
    }
    .dropdown {
        width: 100%;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        font-size: 16px;
        margin-bottom: 10px;
    }
</style>
"""


with gr.Blocks(title="Chat中移小智") as app:
    gr.Markdown(custom_css)

    # 欢迎信息
    gr.Markdown(
        """
        <div class="welcome-message">
            欢迎使用对话式中移小智！
        </div>
        """,
    )

    # 模型切换
    with gr.Tab("精准音色选取"):
        gr.Markdown("### 模型切换", elem_id="tab-title")
        gr.Markdown("#### 如果想体验库以外的音色, 请选取第一选项", elem_id="tab-title")

        with gr.Row():
            # 按钮和下拉菜单的样式
            with gr.Column(scale=4):
                refresh_button = gr.Button("刷新模型", variant="primary", elem_id="refresh-button")

            with gr.Column(scale=4):
                GPT_dropdown = gr.Dropdown(
                    label=i18n("GPT模型选取"),
                    choices=sorted(GPT_names, key=custom_sort_key),
                    value=gpt_path,
                    interactive=True,
                    elem_id="gpt-dropdown",
                    css_class="dropdown"
                )
                SoVITS_dropdown = gr.Dropdown(
                    label=i18n("SoVITS模型选取"),
                    choices=sorted(SoVITS_names, key=custom_sort_key),
                    value=sovits_path,
                    interactive=True,
                    elem_id="sovits-dropdown",
                    css_class="dropdown"
                )

        refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
        SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
        GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

    # 参考音频设置
    with gr.Tab("参考音频"):
        gr.Markdown("### 参考音频设置")
        gr.Markdown("#### 如果没有参考音频，可直接跳至对话", elem_id="tab-title")

        with gr.Row():
            inp_ref = gr.Audio(
                label="上传参考音频（3~10秒内）",
                type="filepath"
            )
            with gr.Column():
                prompt_text = gr.Textbox(
                    label="参考音频的文本",
                    value=""
                )
                prompt_language = gr.Dropdown(
                    label="参考音频的语种",
                    choices=[
                        "中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"
                    ],
                    value="中文"
                )

    # 对话功能
    with gr.Tab("语音对话"):
        with gr.Row():
            with gr.Column():
                # gr.HTML('<div class="custom-button">')
                listen_button = gr.Button("开始对话", variant="primary", elem_id="custom-button")
                # gr.HTML('</div>')

        with gr.Row():
            output = gr.Audio(
                label="中移小智：",
                show_waveform=True
            )

        listen_button.click(
            get_tts_wav,
            [inp_ref, prompt_text, prompt_language],
            [output]
        )

    with gr.Tab("视觉对话"):
        gr.Markdown("### 与中移小智对话")
        with gr.Row():

            media_selection = gr.Radio(choices=["教师", "学生", "医生"], label="选择数字人形象")

            selected_image = gr.Image(label="您好，请提问。", type="filepath")

            gen_video = gr.Video(label="中移小智的回答", format="mp4", type="filepath")

        with gr.Row():

            listen_button = gr.Button("录音并转人机", variant="primary")

        with gr.Row():

            message = gr.Text(label="视频制作状态")

            listen_button.click(
                fn=handle_audio_and_video,
                inputs=[inp_ref, prompt_text, prompt_language],
                outputs=[gen_video, message]
            )

            media_selection.change(
                fn=display_image,
                inputs=media_selection,
                outputs=selected_image
            )


app.queue(concurrency_count=511, max_size=1022).launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=is_share,
    server_port=infer_ttswebui,
    quiet=True,
)
