import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Program that resampled a concentration')
    parser.add_argument("prefix", type=str, help="Prefix of the input file (_conc, _eta1, _eta2) + _<epoch>.bin will be added)")
    parser.add_argument("epoch", type=int, help="Epoch to read from")
    parser.add_argument("N", type=int, help="Number of grid points")
    parser.parse_args()

    ext = ["_conc", "_eta1", "_eta2"]

    for e in ext:
        fname = parser.prefix + e + f"_{parser.epoch}.bin"
        data = np.fromfile(fname, dtype='>f8')
        data = data.reshape((parser.N, parser.N))
        data = data[::2, ::2]
        new_data = np.zeros((parser.N, parser.N))
        half = int(parser.N/2)
        new_data[:half, :half] = data
        new_data[half:, :half] = np.random.rand(half, half)*0.05
        new_data[half:, half:] = np.random.rand(half, half)*0.05
        new_data[:half:, half:] = np.random.rand(half, half)*0.05
        out_fname = parser.prefix + e + f"_{parser.epocj}_renorm.bin"
        new_data.tofile(out_fname, format='>f8')

if __name__ == '__main__':
    main()



    