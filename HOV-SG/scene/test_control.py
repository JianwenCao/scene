import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image

ckpt = "google/siglip2-so400m-patch16-512" 
print(f"Loading {ckpt}...")

model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

print(f"Model Class: {type(model)}")

image = Image.open("cat.jpg").convert("RGB")
texts = ["a cat", "a dog", "a car", "a wall", "nothing"]

inputs = processor(images=image, text=texts, return_tensors="pt", padding="max_length", truncation=True).to(model.device)

with torch.no_grad():
    image_embeds = model.get_image_features(inputs.pixel_values)
    print(f"Image Embeds: shape={image_embeds.shape}, mean={image_embeds.mean():.4f}, std={image_embeds.std():.4f}")
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    text_embeds = model.get_text_features(inputs.input_ids)
    print(f"Text Embeds: shape={text_embeds.shape}, mean={text_embeds.mean():.4f}, std={text_embeds.std():.4f}")
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Cosine similarity
    sims = (image_embeds @ text_embeds.T).squeeze() # (Num_Texts,)

print("\nControl Test Results (SigLIP 2):")
for text, score in zip(texts, sims):
    print(f"{text:<10}: {score.item():.4f}")
