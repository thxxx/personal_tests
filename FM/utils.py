import matplotlib.pyplot as plt

def visualize(img, epoch=0, save=False, chn=3, output_dir="./"):
    if img.shape[0] == 1:
        plt.figure(figsize=(2,2))
        img = img.squeeze()
        if chn==3:
            img = img.permute(1,2,0)
        plt.imshow(img)
        plt.show()
    elif img.shape[0]>1 and len(img.shape)>2:
        fig, axes = plt.subplots(2, 4, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            image = img[i].squeeze()
            if chn==3:
                image = image.permute(1,2,0)
            ax.imshow(image)
            ax.axis('off')

        if save:
            plt.savefig(f'{output_dir}/valid_imgs/valid_{epoch}.png')
        else:
            plt.show()
        plt.close()
    