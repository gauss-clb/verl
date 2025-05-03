import json

def read_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            items.append(json.loads(line))
    return items

def write_jsonl(items, path, mode='w'):
    with open(path, mode, encoding='utf8') as fw:
        for item in items:
            fw.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_text(path):
    items = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            items.append(line.rstrip())
    return items