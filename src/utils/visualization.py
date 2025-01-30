import numpy as np
from paretoset import paretoset
import pandas as pd
import matplotlib.pyplot as plt
import rospy

def get_kerel_size(depth):
    calibP = np.array(rospy.get_param("/theia/right/projection_matrix"))
    P = np.reshape(calibP, (3, 4))  # projection Matrix
    x = depth[:, 0]

def pareto_front(data, plot=False, option=1):
    mask = paretoset(data, sense=["max", "min"], )
    paretoset_sols = data[mask]
    res = np.where(mask == True)[0][0]

    if plot:
        print(mask)
        print(paretoset_sols)
        plt.figure(figsize=(6, 2.5))
        plt.title("Leaves in the Pareto set")
        if option == 0:
            plt.scatter(data[:, 0], data[:, 1], zorder=10, label="All leaves", s=50, alpha=0.8)
            plt.scatter(
                paretoset_sols[:, 0],
                paretoset_sols[:, 1],
                zorder=5,
                label="Optimal leaf",
                s=150,
                alpha=1,
            )
        else:
            plt.scatter(data["dist2minima"], data["dist2maxima"], zorder=10, label="All leaves", s=50, alpha=0.8)
            plt.scatter(
                paretoset_sols["dist2minima"],
                paretoset_sols["dist2maxima"],
                zorder=5,
                label="Optimal leaf",
                s=150,
                alpha=1,
            )

        plt.legend()
        plt.xlabel("dist2minima-[Maximize]")
        plt.ylabel("dist2maxima-[Minimize]")
        plt.grid(True, alpha=0.5, ls="--", zorder=0)
        plt.tight_layout()
        # plt.savefig("example_hotels.png", dpi=100)
        plt.show()
    return res


def calc():
    data_ = pd.DataFrame(  # option = 1
        {'dist2minima': [115, 115, 162, 182, 126, 237, 159, 186, 125, 153, 174, 172, 168, 166, 164, 173,
                         230, 136, 120, 94, 190, 203],
         'dist2maxima': [130, 169, 81, 84, 141, 60, 115, 122, 154, 89, 82, 89, 110, 126, 138, 234,
                         23, 113, 141, 173, 81, 105]})
    data__ = np.array(([115, 115, 162, 182, 126, 237, 159, 186, 125, 153, 174, 172, 168, 166, 164, 173,  # option = 0
                        230, 136, 120, 94, 190, 203],
                       [130, 169, 81, 84, 141, 60, 115, 122, 154, 89, 82, 89, 110, 126, 138, 234,
                        23, 113, 141, 173, 81, 105]))
    print(data__)
    data___ = np.transpose(data__)
    # data___ = np.reshape(data__, (data__.shape[1], data__.shape[0]))
    print(data___.shape)
    res_ = pareto_front(data___, plot=True, option=0)
    print('optimal solution index: ', res_.dtype)
    print('optimal val: ', data___[int(res_), :])


if __name__ == '__main__':
    calc()
