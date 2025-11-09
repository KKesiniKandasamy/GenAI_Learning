import os
from PIL import Image, ImageDraw, ImageFont
class ImageAgent:
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, prompt, size=(512,512)):
        # Hook: replace this method to call a real text->image API (OpenAI, Stable Diffusion, etc.)
        # If no external API keys are configured, generate a placeholder image that displays the prompt.
        filename = os.path.join(self.output_dir, f"image_{abs(hash(prompt))%100000}.png")
        img = Image.new('RGB', size, color=(200,200,230))
        d = ImageDraw.Draw(img)
        # draw prompt text (wrap)
        text = (prompt[:240] + '...') if len(prompt)>240 else prompt
        try:
            font = ImageFont.load_default()
            d.multiline_text((10,10), text, fill=(10,10,10), font=font)
        except Exception:
            d.text((10,10), text)
        img.save(filename)
        return filename
