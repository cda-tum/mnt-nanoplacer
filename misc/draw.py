from PIL import Image, ImageFont
from matplotlib import font_manager

font = font_manager.FontProperties(family='sans-serif', weight='bold')
file = font_manager.findfont(font)
print(file)

text = "Hello world!"
font_size = 36
color = (67, 33, 116, 155)

font = ImageFont.truetype(file, size=font_size)
mask_image = font.getmask(text, "L")
img = Image.new("RGBA", mask_image.size)
img.im.paste(color, (0, 0) + mask_image.size, mask_image)  # need to use the inner `img.im.paste` due to `getmask` returning a core
img.save("yes.png")