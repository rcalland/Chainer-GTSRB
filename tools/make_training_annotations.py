import os
import random
import csv
import fnmatch
from utils import load_config

def write(array, name):
    with open(name, "w+") as f:
        for i in array:
            f.write(i)

def generate_annotations(base_directory, output_file_root, train_file, val_file):

    validation = []
    training = []

    def strip_zeroes(string):
        tmp = string.lstrip("0")
        if tmp == "":
            tmp = "0"
        return tmp

    for _class in sorted(os.listdir(base_directory)):
        class_name = strip_zeroes(_class)
        class_path = os.path.join(base_directory, _class)

        tracks_seen = []
        track_batch = []

        for img in sorted(os.listdir(class_path)):
            if img.endswith(".ppm"):
                track_id = img.split("_")[0]
                track_id = strip_zeroes(track_id)

                image_path = os.path.join(class_path, img)
                annotation = "{} {}\n".format(image_path, class_name)

                track_batch.append([annotation, track_id])

                if not track_id in tracks_seen:
                    tracks_seen.append(track_id)

        # pick a random track number to use as validation
        random.shuffle(tracks_seen)
        val_track_id = tracks_seen[-1]

        for tracks in track_batch:
            if tracks[1] is val_track_id:
                # dont use the oversampled tracks in the validation
                if not fnmatch.fnmatch(tracks[0], "oversampled"):
                    validation.append(tracks[0])
            else:
                training.append(tracks[0])

    write(training, os.path.join(output_file_root, train_file))
    write(validation, os.path.join(output_file_root, val_file))

def generate_test_annotations(csv_file_in, img_directory, txt_file_out):
    annotation = []

    with open(csv_file_in) as f:
        reader = iter(csv.reader(f, delimiter=";"))
        next(reader) # skip the first line
        for row in reader:
            line = "{} {}\n".format(os.path.join(img_directory, row[0]), row[-1])
            #print line

    write(annotation, txt_file_out)

def main():
    cfg = load_config("../config/gtsrb.json")

    generate_annotations("{}/GTSRB/Final_Training/Images".format(cfg["data_root_path"]), "..//annotations", "GTSRB_training.txt", "GTSRB_validation.txt")
    generate_test_annotations("{}/GT-final_test.csv".format(cfg["data_root_path"]), "{}/GTSRB/Final_Test/Images".format(cfg["data_root_path"]), "../GTSRB_test.txt")

if __name__=="__main__":
    main()
