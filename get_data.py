import csv
import numpy as np

FILE_NAME = 'cancer.csv'


def get_data():
    data = []
    with open(FILE_NAME, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
            data.append(lines)

    data = data[1:]

    ground_truth = []
    for r in data:
        if r[1] == 'M':
            ground_truth.append(0)
        else:
            ground_truth.append(1)
    # Converting to numpy array
    ground_truth = np.array(ground_truth, dtype=np.float32)

    data = [row[2:] for row in data]  # deleting columns
    data = np.asarray(data, dtype=np.float32)  # Converting to numpy array
    data = data / np.linalg.norm(data)  # Normalizing
    return data, ground_truth
