import json

def load_config(conf_file):
    with open(conf_file, "r") as f:
        return json.load(f)

def save_config(conf_file, conf_obj):
    with open(conf_file, "w+") as f:
        json.dump(conf_obj, f)
