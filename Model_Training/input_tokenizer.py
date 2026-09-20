def tokenize_function(example, tokenizer):
    return tokenizer(example['text'], truncation = True, max_length = 1024)