import os

nets_selected = ["gan", "wgan", "bgan"]  # gan, wgan, bgan
data_path = "/media/hkuit164/Backup/xjl/20231207_kpsVideo/ml_train_4/train_nobox.csv"
save_dir = "../exp/auto_train_debug"
feature_num = 34
batch_size = 32

labels_path = "/media/hkuit164/Backup/xjl/20231207_kpsVideo/label.txt"

with open(labels_path, 'r') as f:
    actions = f.readlines()

for net_selected in nets_selected:
    for action in actions:
        action = action[:-1]
        cmd = "python {}/{}_train.py --batch_size {} --save_dir {}/{}/{} --action {} --data_path {} --feature_num {}".format(net_selected, net_selected, batch_size, save_dir, net_selected, action, action, data_path, feature_num)
        print(cmd)
        os.system(cmd)
