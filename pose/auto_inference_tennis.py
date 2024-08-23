import os

nets_selected = ["gan", "wgan", "bgan"]
data_folder = "/media/hkuit164/Backup/PoseAndSceneGenerator/exp_tennis/2024-08-12"
action = "right"

for net_selected in nets_selected:
    # print(os.path.exists("/media/hkuit164/Backup/PoseAndSceneGenerator/pose/wgan/wgan_inference.py"))
    gan_folder = os.path.join(os.path.join(data_folder, net_selected+"_"+action))
    cmd = "python /media/hkuit164/Backup/PoseAndSceneGenerator/pose/{}/{}_inference.py --generator_path {} --csv_path {}".format(net_selected, net_selected, os.path.join(gan_folder, "generator.pth"), os.path.join(gan_folder, "generated.csv"))
    print(cmd)
    os.system(cmd)
