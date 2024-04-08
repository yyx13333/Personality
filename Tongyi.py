import csv
from http import HTTPStatus
from typing import Dict

import dashscope
from dashscope.api_entities.dashscope_response import (GenerationResponse,
                                                       Message, Role)


# TOKEN::sk-712d35327a2c42339d43daebf8e63f2b

def sample_sync_call():
    prompt_text = '玛丽离婚了，请将这句话换一种形式说出来'
    resp = dashscope.Generation.call(
        model='qwen-turbo',
        prompt=prompt_text
    )
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if resp.status_code == HTTPStatus.OK:
        print(resp.output)  # The output text
        print(resp.usage)  # The usage information
    else:
        print(resp.code)  # The error code.
        print(resp.message)  # The error message.


def sample_sync_call_streaming():
    prompt_text = '用萝卜、土豆、茄子做饭，给我个菜谱。'
    response_generator = dashscope.Generation.call(
        model='qwen-turbo',
        prompt=prompt_text,
        stream=True,
        top_p=0.8)

    head_idx = 0
    for resp in response_generator:
        paragraph = resp.output['text']
        print("\r%s" % paragraph[head_idx:len(paragraph)], end='')
        if (paragraph.rfind('\n') != -1):
            head_idx = paragraph.rfind('\n') + 1



prompt = '你是一个擅长挖掘对话中心理特征信息的对话助手，下面是一段对话，将说话者说的话语情绪增强，但是却要符合对话原有的意思”'
def call_with_messages(myStr):
    one = {'role': 'system',
           'content': prompt}
    messages = [one]
    dict = {'role': 'user', 'content': myStr}
    messages.append(dict)
    print(messages)
    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
        # print(response.output.choices[0].message.content)
        return  response.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

def semanticExtension():
    with open("./myCPEDTongyi.csv","w",encoding="utf-8") as p:
        with open("../data/CPED/train_split.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            writer = csv.writer(p)
            count = 0
            flag = 0
            dialogue_ID = ''
            newRow = []
            myStr = ''
            for row in reader:
                if flag == 0:
                    flag = 1
                    continue
                #---------------------------
                if dialogue_ID == '':
                    dialogue_ID = row[1]
                if row[1] != dialogue_ID:
                    if count <= 60:
                        t = call_with_messages(myStr)
                    if t == '':
                        t = 'None'
                    newRow.append(dialogue_ID)
                    newRow.append(t)
                    writer.writerow(newRow)
                    newRow = []
                    count = 0
                    print(t)
                    myStr = ''
                    myStr += row[3]+'以一种'+row[15]+'的情绪说:'+row[17]+'。'
                else:
                    count += 1
                    myStr += row[3] +'以一种' + row[15] + '的情绪说:' + row[17] + '。'

            t = call_with_messages(myStr)
            newRow.append(dialogue_ID)
            newRow.append(t)
            writer.writerow(newRow)
        f.close()
    p.close()

if __name__ == '__main__':
    # call_with_messages('可以的！')
    semanticExtension()
