import pickle
import os
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from feature_extraction import extract_features
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")


cwd = os.path.join(os.getcwd(), "data")
source   = os.path.join(cwd, "dataset")
modelpath = os.path.join(cwd, "training")
training_file = os.path.join(modelpath, "train_data.txt")
test_file = os.path.join(modelpath, "test_data.txt")

cepstra = (24,)
window = (0.02, 0.025, 0.03)
dist_window = (0.005, 0.01, 0.015, 0.02)
length = len(cepstra) * len(window) * len(dist_window)


file_paths = open(training_file,'r')

for l in tqdm(range(length)):
    d = dist_window[l%4]
    w = window[(l//4)%3]
    c = cepstra[l//12]
    print("\nMFCCs: {} / window: {} / distance: {}".format(c, w, d))
    count = 1

    features = np.asarray(())
    for path in file_paths:
        path = path.strip()

        sr, audio = read(os.path.join(source, path))
        vector = extract_features(audio, sr, window=w, dist_window=d, cepstra=c)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        if count == 5:
            gmm = GaussianMixture(n_components=c,
                                    max_iter=200,
                                    covariance_type='diag',
                                    n_init=3)
            gmm.fit(features)
            picklefile = path.split("-")[0]+".gmm"
            pickle.dump(gmm, open(os.path.join(modelpath, picklefile), 'wb'))
            features = np.asarray(())
            count = 0
        count = count + 1


    file_paths = open(test_file, 'r')

    gmm_files = [os.path.join(modelpath, fname) for fname in 
                os.listdir(modelpath) if fname.endswith('.gmm')]
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
                in gmm_files]

    for path in file_paths:
        path = path.strip()
        sr, audio = read(os.path.join(source, path))
        vector   = extract_features(audio, sr, window=w, dist_window=d, cepstra=c)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = np.argmax(log_likelihood)
        print(path, ",", speakers[winner])
        time.sleep(1.0)
