from shml.interface import Infer
model_path = "/gemini/code"
infer_instance = Infer(model_path)
image_path = "/gemini/code/1.png"
from PIL import Image
image = Image.open(image_path)
threshold = 0.68
image_size = 960
keep_ratio = True

space = True
escape = True
conf = True

# Perform inference
result, result_dict = infer_instance.infer_one(
    img=image,
    threshold=threshold,
    image_size=image_size,
    keep_ratio=keep_ratio,
    model_path=model_path,
    space=space,
    escape=escape,
    conf=conf
)
print(result, result_dict)
