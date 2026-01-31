import json



def read_json(file_name:str):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data