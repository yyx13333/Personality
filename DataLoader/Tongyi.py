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



prompt = '用于提示生成预言性常识推理的大型语言模型的模板输入如下：给定听者和说话者之间的二元对话片段，' \
         '目标是理解对话并进行推理以识别说话者内部的心理(对听者陈述的话语起作用的原因)。' \
         '”'
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
    with open("./myCPEDTongyi.csv","w",encoding="utf-8" ,newline='') as p:
        with open("../../data/myCPED/myCped_clearing_0_200.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            writer = csv.writer(p)
            count = 0
            flag = 0
            lunci = 0
            newRow = []
            myStr = ''
            for row in reader:
                if flag == 0:
                    flag = 1
                    continue
                #----------说明是新一轮对话-------------
                if lunci > int(row[1]):
                    if lunci < 60:
                        t = call_with_messages(myStr)
                        if t == '':
                            print("违规内容:"+myStr)
                        else:
                            writer.writerow(t)
                    newRow = []
                    myStr = ''
                    lunci = int(row[1])
                    myStr += row[0] + '以一种' + row[2] + '的情绪说:' + row[3] + '。'
                else:
                    lunci = int(row[1])
                    myStr += row[0] + '以一种' + row[2] + '的情绪说:' + row[3] + '。'
            t = call_with_messages(myStr)
            writer.writerow(t)
        f.close()
    p.close()

if __name__ == '__main__':
    semanticExtension()



