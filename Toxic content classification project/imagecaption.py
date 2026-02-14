from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_caption(image):

    inputs = processor(images=image, return_tensors="pt").to(device)

    output = model.generate(**inputs, max_length=50)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption
