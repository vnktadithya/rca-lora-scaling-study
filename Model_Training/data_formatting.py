def format_instruction(example):
    messages = example['messages']
    system_content = messages[0]['content']
    user_content = messages[1]['content']
    assistant_content = messages[2]['content']

    text = f'<s>[INST] {system_content}\n\n{user_content} [/INST] {assistant_content}</s>'
    return {'text': text}

