import os
import argparse
from tqdm import tqdm
import requests
import zipfile
from utils import load_config, save_config

gtsrb_url = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
gtsrb_test_url = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
gtsrb_test_annotations_url = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"

def download(url, local_path):
    # download training data
    response = requests.get(url, stream=True)
    local_path = os.path.join(local_path, os.path.basename(url))
    with open(local_path, "wb") as handle:
        for data in tqdm(response.iter_content(chunk_size=1024)):
            if data:
                handle.write(data)
    return local_path

def unzip(local_file):
    data_path = os.path.dirname(os.path.abspath(local_file))
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()
    return data_path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-o", type=str, required=True)
    parser.add_argument("--data_url", "-i", type=str, default=gtsrb_url)
    return parser.parse_args()

def set_config(config_file, base_dir):
    cfg = load_config(config_file)
    cfg["data_root_path"] = base_dir
    save_config(config_file, cfg)

def main():
    args = parse_arguments()
    os.system("mkdir -p {}".format(args.data_dir))

    # training data
    local_file = download(args.data_url, args.data_dir)
    base_dir = unzip(local_file)

    # test data
    local_file = download(gtsrb_test_url, args.data_dir)
    base_dir = unzip(local_file)

    # test data annotations
    local_file = download(gtsrb_test_annotations_url, args.data_dir)
    base_dir = unzip(local_file)
    
    # update config file
    set_config("../config/gtsrb.json", base_dir)

if __name__=="__main__":
    main()
