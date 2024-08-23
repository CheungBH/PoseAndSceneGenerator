import os
from datetime import datetime

current_date = datetime.now()
formatted_date = current_date.strftime('%Y-%m-%d')
nets_selected = ["gan", "wgan", "bgan", "cgan"]
data_path = "/home/hkuit164/Downloads/right.csv"
save_dir = "exp_tennis/{}".format(formatted_date)
feature_num = 34
batch_size = 32
action = "right"

# labels_path = "/media/hkuit164/Backup/xjl/20231207_kpsVideo/label.txt"
#
# with open(labels_path, 'r') as f:
#     actions = f.readlines()

for net_selected in nets_selected:
    if os.path.exists("pose/"+net_selected+'/'+net_selected+'_train.py') == True:
        cmd = "python pose/{}/{}_train.py --batch_size {} --save_dir {}/{} --data_path {} --feature_num {}".format(net_selected, net_selected, batch_size, save_dir, net_selected+"_"+action, data_path, feature_num)
        print(cmd)
        os.system(cmd)
