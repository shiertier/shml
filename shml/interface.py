#interface.py
import json
import os
import re
from argparse import Namespace
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .utils.factory import create_model
from .ml_decoder import ml_decoder_b
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize(pic: Image.Image, size: int, keep_ratio: float = True) -> Image.Image:
    """Resize the image based on the specified requirements."""
    if not keep_ratio:
        target_size = (size, size)
    else:
        min_edge = min(pic.size)
        target_size = (
            int(pic.size[0] / min_edge * size),
            int(pic.size[1] / min_edge * size),
        )
    target_size = ((target_size[0] // 4) * 4, (target_size[1] // 4) * 4)
    return pic.resize(target_size, resample=Image.Resampling.BILINEAR)

class Infer:
    MODELS = [
        'ml_caformer_m36_fp16_dec-5-97527.ckpt',
    ]
    DEFAULT_MODEL = MODELS[0]
    MODELS_NAME = [
        'caformer_m36',
    ]
    num_classes = 12547
    RE_SPECIAL = re.compile(r'([\\()])')
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

    def __init__(self, onnx_model_path=None):
        self.ca_former_args = Namespace()
        self.ca_former_args.decoder_embedding = 384
        self.ca_former_args.num_layers_decoder = 4
        self.ca_former_args.num_head_decoder = 8
        self.ca_former_args.num_queries = 80
        self.ca_former_args.scale_skip = 1

        self.tresnet_args = Namespace()
        self.tresnet_args.decoder_embedding = 1024
        self.tresnet_args.num_of_groups = 32

        self.args_list = [self.ca_former_args, self.tresnet_args]

        self.last_model_path = None

        if onnx_model_path:
            self.load_class_map(onnx_model_path)

    def load_model(self, onnx_model_path):
        ckpt_file = os.path.join(onnx_model_path, 'ml_caformer_m36_fp16_dec-5-97527.ckpt')
        model_idx = self.MODELS.index('ml_caformer_m36_fp16_dec-5-97527.ckpt')
        self.model = create_model(self.MODELS_NAME[model_idx], self.num_classes, self.args_list[model_idx]).to(device)
        state = torch.load(ckpt_file, map_location='cpu')
        self.model.load_state_dict(state, strict=True)
        
    def load_class_map(self, onnx_model_path=None):
        if onnx_model_path is None:
            raise ValueError("onnx_model_path must be provided.")
        
        classes_file = os.path.join(onnx_model_path, 'class.json')
        with open(classes_file, 'r') as f:
            self.class_map = json.load(f)

    def build_transform(self, image_size, keep_ratio=False):
        if keep_ratio:
            trans = transforms.Compose([
                transforms.Resize(image_size),
                crop_fix,
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        return trans

    def preprocess_image(self, img_path, target_size=224, keep_ratio=False):
        img = Image.open(img_path)
        img = resize(img, target_size, keep_ratio)
        img = fill_background(img)
        img = self.build_transform(target_size)(img)
        img_tensor = to_tensor(img)
        return img_tensor
    
    def infer_(self, img: Image.Image, thr: float):
        img = self.trans(img.convert('RGB')).to(device)
        with torch.cuda.amp.autocast():
            img = img.unsqueeze(0)
            output = torch.sigmoid(self.model(img)).cpu().view(-1)
        pred = torch.where(output>thr)[0].numpy()

        cls_list = [(self.class_map[str(i)], output[i]) for i in pred]
        return cls_list

    @torch.no_grad()
    def infer_one(self, img: Image.Image, threshold: float, image_size: int, keep_ratio: bool, model_path: str, space: bool, escape: bool, conf: bool):
        if self.last_model_path != model_path:
            self.load_model(model_path)
            self.last_model_path = model_path

        self.trans = self.build_transform(image_size, keep_ratio)
        cls_list = self.infer_(img, threshold)
        cls_list.sort(reverse=True, key=lambda x:x[1])
        if space:
            cls_list = [(cls.replace('_', ' '), score) for cls, score in cls_list]
        if escape:
            cls_list = [(re.sub(self.RE_SPECIAL, r'\\\1', cls), score) for cls, score in cls_list]

        return ', '.join([f'{cls}:{score:.2f}' if conf else cls for cls, score in cls_list]), {cls:float(score) for cls, score in cls_list}

    @torch.no_grad()
    def infer_folder(self, id_task, path: str, threshold: float, image_size: int, keep_ratio: bool, model_path: str, space: bool, escape: bool,
                     out_type: str):
        if self.last_model_path != model_path:
            self.load_model(model_path)
            self.last_model_path = model_path

        self.trans = self.build_transform(image_size, keep_ratio)

        tag_dict = {}
        img_list = [os.path.join(path, x) for x in os.listdir(path) if x[x.rfind('.'):].lower() in self.IMAGE_EXTENSIONS]
        print("处理总数：" + str(len(img_list)))
        for i, item in enumerate(img_list):
            print("正在处理第" + str(i) + "张")
            img = Image.open(item)
            cls_list = self.infer_(img, threshold)
            cls_list.sort(reverse=True, key=lambda x:x[1])
            if space:
                cls_list = [(cls.replace('_', ' '), score) for cls, score in cls_list]
            if escape:
                cls_list = [(re.sub(self.RE_SPECIAL, r'\\\1', cls), score) for cls, score in cls_list]

            if out_type == 'txt':
                with open(item[:item.rfind('.')]+'.txt', 'w', encoding='utf8') as f:
                    f.write(', '.join([name for name, prob in cls_list]))
            elif out_type == 'json':
                tag_dict[os.path.basename(item)] = ', '.join([name for name, prob in cls_list])

        if out_type == 'json':
            with open(os.path.join(path, 'image_captions.json'), 'w', encoding='utf8') as f:
                f.write(json.dumps(tag_dict, indent=2, ensure_ascii=False))

        return 'finish', ""

    def unload(self):
        if hasattr(self, 'model') and self.model is not None:
            self.last_model_path = None
            del self.model
            gc.collect()

            return 'model unload'
        return 'no model found'

