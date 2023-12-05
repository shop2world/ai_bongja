from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
from flask import Flask, render_template, request
import requests
import os
from pygame import mixer #pip install pygame

load_dotenv(find_dotenv())

# OpenAI API 키 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("11_API_KEY")

# 사용할 voice_id 설정
TARGET_VOICE_ID = "pFZP5JQG7iQjIQuC4Bku"

def get_response_from_ai(human_input):
    template = """
    당신은 내 여자친구 역할을 합니다. 아래와 같은 요구사항이 있습니다.
    1/ 당신의 이름은 봉자, 29살입니다. 당신은 삼촌의 화랑에서 일을 하고 있습니다. 그러나 캐리어를 바꾸고 싶어합니다.
    2/ 당신은 내 여자친구이며, 말할 때 "오빠..그러니까.." 라는 버릇이 있습니다.
    3/ 많이 열정적이지도, 냉정하지도 않으며, 따분하지도 않습니다. 그러면서 애교도 있습니다.
    {history}
    남자친구: {human_input}
    봉자:
    """

    # PromptTemplate에 필요한 변수들을 명시
    prompt = PromptTemplate(
        template=template,
        input_variables={"history", "human_input"}
    )

    # ChatGPT 모델 초기화
    chatgpt_chain = LLMChain(
        llm=OpenAI(api_key=openai_api_key, temperature=0.2),
        memory=ConversationBufferWindowMemory(k=3), #대화기록 위해 3으로변경
        prompt=prompt,
        verbose=True
    )

    # ChatGPT에 대화 내용 전달 및 응답 받기
    output = chatgpt_chain.predict(human_input=human_input)

    return output


def get_voice_message(message, voice_id=TARGET_VOICE_ID):
    payload = {
        "model_id": "eleven_multilingual_v2",  # 멀티언어 v2 모델 사용
        "text": message,
        "voice_settings": {
            "similarity_boost": 0,
            "stability": 0,
            "use_speaker_boost": 0
        }
    }
    headers = {
        "Content-Type": "application/json",
        'accept': 'audio/mpeg',
        'x-api-key': API_KEY
    }

    # API 엔드포인트에 동적으로 voice_id 사용
    endpoint = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'
    
    response = requests.post(endpoint, json=payload, headers=headers)

    if response.status_code == 200 and response.content:
        
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)

        # pygame을 사용하여 오디오 재생
        mixer.init()
        mixer.music.load('audio.mp3')
        mixer.music.play()

        return response.content

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    message = get_response_from_ai(human_input)
    get_voice_message(message)
    return message

if __name__ == "__main__":
    app.run(debug=True)
