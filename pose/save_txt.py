import os

def save_data(epoch, D_loss, G_loss, save_path, last=False):
    with open(os.path.join(save_path, "result.txt"), "a") as f:
        # if last:
        #     f.write("Total epoch: {}, Average D_loss: {}, Average G_loss: {}".format(epoch, D_loss, G_loss))
        # else:
        f.write("Epoch: {}, D_loss: {}, G_loss: {}\n".format(epoch, D_loss, G_loss))

