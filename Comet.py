from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig




if __name__ == "__main__":
    config = AutoConfig.from_pretrained("../../comet/config.json")
    model = AutoModelForSeq2SeqLM.from_pretrained("../../comet/pytorch_model.bin", config=config)
    tokenizer = AutoTokenizer.from_pretrained("../../comet/",
                                              config="../../comet/tokenizer_config.json",
                                              vocab_file="../../comet/vocab.json")

    # 准备输入文本
    input_text2 = "{你真好 [GEN] oReact"
    # 使用tokenizer对输入文本进行编码
    input_ids = tokenizer(input_text2, return_tensors="pt", truncation=True, padding="max_length").input_ids
    print(input_ids.shape)
    relation_set = ["oEffect", "oReact", "xEffect", "xReact"]
    summaries = model.generate(
        input_ids=input_ids, max_length=100, num_beams=6, early_stopping=True, num_return_sequences=6
        )

    # 解码模型的输出
    decoded_output2 = tokenizer.decode(summaries[0], skip_special_tokens=True)
    decoded_output3 = tokenizer.decode(summaries[1], skip_special_tokens=True)
    decoded_output4 = tokenizer.decode(summaries[2], skip_special_tokens=True)
    decoded_output5 = tokenizer.decode(summaries[3], skip_special_tokens=True)
    decoded_output6 = tokenizer.decode(summaries[4], skip_special_tokens=True)
    print(decoded_output2)
    print(decoded_output3)
    print(decoded_output4)
    print(decoded_output5)
    print(decoded_output6)
    print(summaries)
