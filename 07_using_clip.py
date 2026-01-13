#%%

from transformers import CLIPProcessor, CLIPModel
from PIL import Image

#%%

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# zero shot classification

image = Image.open('X')
queries = ['test']

inputs = processor(text=queries, images=image, return_tensors='pt', padding=True)
ouputs = model(**inputs)

# get similarity scores
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print("Predictions: ")
for text,prob in zip(queries, probs[0]):
    print(f"    {text}: {prob:.3f}")