from openai import OpenAI
import os
import pdb

def get_response():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model="qwen2-math-72b-instruct",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$.'}]
        )
    pdb.set_trace()
    a = completion.model_dump_json()
    print(completion.model_dump_json()["choices"][0]["messgae"]["content"])

if __name__ == '__main__':
    get_response()