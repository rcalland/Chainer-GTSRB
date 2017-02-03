import json
from utils import load_config
from skimage import io
import numpy as np

def get_stats(filename):
    img = io.imread(filename)
    return img.mean(), img.std()

def sample_mean(filename):
    with open(filename, "r") as f:
        muarray, sigmaarray = zip(*[get_stats(x.split(" ")[0]) for x in f])
        avg_mu = np.mean(muarray)
        #avg_sigma = np.mean(sigmaarray)
        return avg_mu

def calc_mean(config_file):
    config = load_config(config_file)
    train_mean = sample_mean(config["train_annotation"])
    val_mean = sample_mean(config["validation_annotation"])
    
    config["train_mean"] = train_mean
    config["validation_mean"] = val_mean

    with open(config_file, "w+") as f:
        json.dump(config, f)

def main():
    config_files = ["../config/gtsrb.json"]
    
    for i in config_files:
        calc_mean(i)

if __name__=="__main__":
    main()
