from scipy.io import loadmat 
import pandas as pd
import numpy as np


def decode_mat(file_path):
    data = loadmat(file_path)
    data = {k:v for k, v in data.items() if k[0] != '_'}

    for key, value in data.items():
        df = pd.DataFrame(value)

        df.columns = [i for i in np.linspace(1000, 12000, value.shape[1])]
        df.index = [i for i in np.linspace(0, 360, value.shape[0])]
        
    return df

if __name__ == "__main__":
    df = decode_mat(r"/workspace/sphere_metasurface/mat_data/DJI_el0deg_RCS.mat")
    print(df.iloc[:5, 600:610])