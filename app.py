import gradio as gr
from PIL import Image
import torch
from torchvision import transforms

from colornet.model import ColorizationUNet


def denormalize(tensor):
    if torch.max(tensor) <= 1:
        return tensor * 255
    return tensor


def colorize(input, model):
    N = 256
    transform_bw = transforms.Compose(
        [
            transforms.Grayscale(),  # Convert to grayscale if needed
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    image = transform_bw(input)
    image = image.view(1, 1, N, N)
    with torch.inference_mode():
        tensor = model(image)
    tensor = denormalize(tensor[0])
    tensor = tensor.permute(1, 2, 0)  # Convert from 3xNxN to NxNx3
    tensor = tensor.cpu().numpy().astype("uint8")
    image = Image.fromarray(tensor)
    return image


model = ColorizationUNet(mean=0.4541, std=0.2502)
model.load_state_dict(torch.load("build/train/2024-01-02_23-04-02/model_weights.pt"))
demo = gr.Interface(
    fn=lambda x: colorize(x, model), inputs=gr.Image(type="pil"), outputs="image"
)

if __name__ == "__main__":
    demo.launch(show_api=False)
