from torchvision import transforms
from PIL import Image

def preprocess_image(image):
    image = Image.open(image)
    transform=transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image

