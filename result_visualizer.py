import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_log(log_file,mode = 'default'):
    loaded_data = []
    img_ids = []
    sp_features = []
    norm_features = []
    sp_points = []
    norm_points = []
    gt_points = []
    euler_rpy = []
    if mode == '2_rpy':
        euler_rpy_norm = []
    with open(log_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            try:
                tmp_data = line.split()

                img_ids.append(int(tmp_data[0]))
                sp_features.append(int(tmp_data[1]))
                norm_features.append(int(tmp_data[2]))
                sp_points.append([float(x) for x in tmp_data[3:6]])
                norm_points.append([float(x) for x in tmp_data[6:9]])
                gt_points.append([float(x) for x in tmp_data[9:12]])
                euler_rpy.append([float(x) for x in tmp_data[12:15]])
                if mode == '2_rpy':
                    euler_rpy_norm.append([float(x) for x in tmp_data[15:18]])
            except ValueError:
                print(tmp_data)
                print("\nfailed")
    if mode == "default":
        return np.array(img_ids), np.array(sp_features), np.array(norm_features), np.array(sp_points), np.array(norm_points), np.array(gt_points),np.array(euler_rpy)
    elif mode == "2_rpy":
        return np.array(img_ids), np.array(sp_features), np.array(norm_features), np.array(sp_points), np.array(norm_points), np.array(gt_points),np.array(euler_rpy),np.array(euler_rpy_norm)
    

def main():
    # dataloader
    img_ids, sp_features, norm_features, sp_points, norm_points, gt_points, euler_rpy,euler_rpy_norm= read_log(
        "results/DSEC_test_save.txt",mode="2_rpy")
    _, _, _, _, _, _, euler_rpy_gt= read_log(
        "results/DESC_interlaken_00_c_save.txt")
    euler_rpy_gt = euler_rpy_gt[1:-1]

    # error
    sp_error = np.linalg.norm((sp_points - gt_points)[:, [0, 2]], axis=1)
    norm_error = np.linalg.norm((norm_points - gt_points)[:, [0, 2]], axis=1)
    # average error
    avg_sp_error = [np.mean(sp_error[:i]) for i in range(len(sp_error))]
    avg_norm_error = [np.mean(norm_error[:i]) for i in range(len(norm_error))]

    print("SuperPoint : ", avg_sp_error[-1])
    print("Normal     : ", avg_norm_error[-1])

    # visualize
    figure = plt.figure()
    if True:
        # plt.subplot(2, 1, 1)
        plt.plot(img_ids, norm_features, color="blue", label="Normal-VO")
        plt.plot(img_ids, sp_features, color="red", label="Eventpoint-VO")
        plt.ylabel("Feature Number")
        plt.xlabel("steps")
        plt.legend(["Normal-VO","Eventpoint-VO"])
        plt.savefig("feature_num.png")
        
        # plt.subplot(2, 1, 2)
        # plt.plot(img_ids, avg_norm_error, color="blue", label="Normal-VO")
        # plt.plot(img_ids, avg_sp_error, color="red", label="SP-VO")
        # plt.xlabel("Timestamp")
        # plt.ylabel("Avg Distance Error [m]")
        # plt.legend()

        plt.subplot(3,1,1)
        plt.plot(img_ids,euler_rpy_norm[:,0],color="blue", label="Normal-VO")
        plt.plot(img_ids,euler_rpy[:,0],color="red", label="Eventpoint-VO")
        plt.plot(img_ids,euler_rpy_gt[:,0],color="green", label="gt")
        plt.legend()
        plt.ylabel("theta_x")
        plt.xlabel("steps")

        plt.subplot(3,1,2)
        plt.plot(img_ids,euler_rpy_norm[:,1],color="blue", label="Normal-VO")
        plt.plot(img_ids,euler_rpy[:,1],color="red", label="Eventpoint-VO")
        plt.plot(img_ids,euler_rpy_gt[:,1],color="green", label="gt")
        plt.ylabel("theta_y")
        plt.xlabel("steps")

        plt.subplot(3,1,3)
        plt.plot(img_ids,euler_rpy_norm[:,2],color="blue", label="Normal-VO")
        plt.plot(img_ids,euler_rpy[:,2],color="red", label="Eventpoint-VO")
        plt.plot(img_ids,euler_rpy_gt[:,2],color="green", label="gt")
        plt.ylabel("theta_z")
        plt.xlabel("steps")
        plt.savefig("rotation_euler.png")


    else:
        plt.plot(gt_points[:, 0], gt_points[:, 2],
                 color="black", label="Ground Truth")
        plt.plot(norm_points[:, 0], norm_points[:, 2],
                 color="blue", label="Normal-VO")
        plt.plot(sp_points[:, 0], sp_points[:, 2], color="red", label="SP-VO")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend()
    plt.show()
    print("finish")

if __name__ == "__main__":
    main()
