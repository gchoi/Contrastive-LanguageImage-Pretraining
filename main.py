# %% Import packages
import os
import clip
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision.datasets import CIFAR100


def configurations() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = "ViT-B/32"

    configs = {
        "device": device,
        "clip_model": clip_model
    }

    return configs


def load_model(configs: dict):
    print(clip.available_models())

    model, preprocess = clip.load(
        name=configs['clip_model'],
        device=configs['device'],
        download_root="./models/"
    )

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    return model, preprocess


def image_and_texts(preprocess: torchvision.transforms.transforms.Compose):
    # images in skimage to use and their textual descriptions
    descriptions = {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse",
        "coffee": "a cup of coffee on a saucer"
    }

    original_images = []
    images = []
    texts = []
    plt.figure(figsize=(16, 5))

    for filename in [
        filename for filename in os.listdir(skimage.data_dir) if
        filename.endswith(".png") or filename.endswith(".jpg")
    ]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue

        image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")

        plt.subplot(2, 4, len(images) + 1)
        plt.imshow(image)
        plt.title(f"{filename}\n{descriptions[name]}")
        plt.xticks([])
        plt.yticks([])

        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descriptions[name])

    plt.tight_layout()
    plt.show()
    return descriptions, original_images, images, texts


def build_features(
    configs: dict,
    model: clip.model.CLIP,
    images: list,
    texts: list
):
    image_input = torch.tensor(np.stack(images)).to(configs['device'])
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).to(configs['device'])
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
    return image_features, text_features


def calculate_cosine_similarity(
    descriptions: dict,
    original_images: list,
    texts: list,
    image_features: torch.Tensor,
    text_features: torch.Tensor
):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    count = len(descriptions)

    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
    plt.show()
    return


def zeroshot_image_classification(
    configs: dict,
    preprocess: torchvision.transforms.transforms.Compose,
    model: clip.model.CLIP,
    original_images: list,
    image_features: torch.Tensor
):
    cifar100 = CIFAR100(
        "./dataset/",
        transform=preprocess,
        download=True
    )
    text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
    text_tokens = clip.tokenize(text_descriptions).to(configs['device'])

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

    plt.figure(figsize=(16, 16))

    for i, image in enumerate(original_images):
        plt.subplot(4, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [cifar100.classes[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.show()
    return


# %% main
def main():
    configs = configurations()

    ## 1. Load the CLIP model
    model, preprocess = load_model(configs)
    # check preprocess
    print(preprocess)


    ## 2. Set up input images and texts
    descriptions, original_images, images, texts = image_and_texts(preprocess)


    ## 3. Build features
    image_features, text_features = build_features(
        configs=configs,
        model=model,
        images=images,
        texts=texts
    )


    ## 4. Calculate cosine similarity
    calculate_cosine_similarity(
        descriptions=descriptions,
        original_images=original_images,
        texts=texts,
        image_features=image_features,
        text_features=text_features
    )


    ## 5. Zero-Shot Image Classification
    zeroshot_image_classification(
        configs=configs,
        preprocess=preprocess,
        model=model,
        original_images=original_images,
        image_features=image_features
    )


if __name__ == '__main__':
    configs = configurations()
    main()