import pandas as pd

from shapelet import *
if __name__ == "__main__":
    # df = pd.read_csv("test.csv")
    print(gen_sampled_shapes([pd.DataFrame([1,2,3,4,5]),[2,3,4,5,6],[3,4,5,6,7]],num_shapes=1))