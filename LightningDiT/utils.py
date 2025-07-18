import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# 예: image shape = (3, H, W), 값은 [0, 1]
def show_tensor_image(img_tensor):
    # tensor → numpy
    img = img_tensor.detach().cpu()
    if img.ndim == 4:  # 배치가 있을 경우 첫 번째 이미지만
        img = img[0]
    img = F.to_pil_image(img)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

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


def count_parameters(model, only_trainable: bool = True):
    return f"{round(sum(p.numel() for p in model.parameters() if p.requires_grad or not only_trainable)/1000000, 3)}M"
