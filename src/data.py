# data.py
# process (input) data for MANTA

import pandas as pd
import numpy as np
import os.path

def get_patients (file_name="patients.csv", datadir="../data"):

    file_path = os.path.join(datadir, file_name)

    patients = pd.read_csv(file_path, sep=";")

    return patients

def get_biomarkers (file_name="biomarkers.csv", datadir="../data"):

    file_path = os.path.join(datadir, file_name)

    biomarkers = np.genfromtxt(file_path, delimiter=";")

    return biomarkers
