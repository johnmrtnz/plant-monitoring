import base64
import anthropic
from PIL import ImageOps
from PIL import Image
import io

IMG_PATH = '/Users/johnmartinez/Documents/repo/plant-monitoring/data/IMG_5835.jpeg'
IMG_TYPE = 'image/jpeg'

# ── Resize & compress to stay under the 5MB base64 limit ──────────────────
def prepare_image(path: str, max_size: tuple = (1568, 1568), quality: int = 85) -> str:
    """
    Resize and compress an image, then return as a base64 string.
    1568px is the sweet spot Claude recommends for image analysis — 
    large enough for detail, small enough to be cost-efficient.
    """
    with Image.open(path) as img:
        # Preserve orientation from EXIF data (iPhone photos often need this)
        img = ImageOps.exif_transpose(img)
        
        # Convert RGBA/palette images to RGB (JPEG doesn't support transparency)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # Resize — thumbnail() preserves aspect ratio and only shrinks, never upscales
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Write to an in-memory buffer (no temp files needed)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        
        return base64.standard_b64encode(buffer.read()).decode("utf-8")

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": IMG_TYPE,
                        "data": prepare_image(IMG_PATH),
                    },
                },
                {"type": "text", "text": "What kind of seedlings am I growing here? And are they healthy?"},
            ],
        }
    ],
)
print(message)