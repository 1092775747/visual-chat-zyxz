import speech_recognition as sr
from aip import AipSpeech
import pyaudio
import json
import requests

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

assistant_id = "669f241dcd4cd414b120364a"

""" 你的 APPID AK SK """
# 百度申请，标准段语音即可，个人可以免费体验
APP_ID = '106289181'
API_KEY = 'FDuXkziOm1LChASa52n4WZ1E'
SECRET_KEY = 'PswuORMmPTd9Z7rDpkSbOitIGGuJAHht'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)



def speech_to_text(max_audio_time = 8):
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
        print('您说的是：', text['result'][0])
        return text['result'][0]

if __name__ == '__main__':
    p = pyaudio.PyAudio()
    token = get_access_token(api_key, api_secret)


    devices = p.get_device_count()
    for i in range(devices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 1:
            print(f"Microphone: {device_info.get('name')} , Device Index: {device_info.get('index')}")

    
    access_token = token
    for i in range(3):
        ask_text = speech_to_text(6)
        print('\n用户:', ask_text,'\n')
        result = send_message(assistant_id, access_token, ask_text)
        if result != None:
            print(result)