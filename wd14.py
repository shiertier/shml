import argparse
import csv
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import zipfile

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
try:
    import pillow_avif
    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass
try:
    from jxlpy import JXLImagePlugin
    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass
try:
    import pillow_jxl
    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

def glob_images_pathlib(dir_path):
    """
    递归地查找指定目录下的图像文件，并返回这些图像文件的路径列表。
    """
    dir_path = Path(dir_path)
    image_paths = []  # 创建一个空列表，用于存储图像文件的路径
    for ext in IMAGE_EXTENSIONS:  # 对于每个图像文件扩展名
        image_paths += list(dir_path.rglob("*" + ext))  # 使用rglob方法递归查找匹配该扩展名的文件路径，并添加到image_paths列表中
    image_paths = list(set(image_paths))  # 将image_paths列表转换为集合，去除重复的路径，然后再转换回列表
    image_paths.sort()  # 对图像路径列表进行排序
    return image_paths  # 返回最终的图像文件路径列表1

# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

def preprocess_image(image):
    """
    对输入的图像进行预处理，包括转换通道顺序、填充成正方形、调整大小并转换数据类型。
    """
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)
    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    image = image.astype(np.float32)
    return image

class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    """
    加载、预处理和准备图像数据。
    """
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            tensor = torch.tensor(image)
        except Exception as e:
            print(f"无法加载图片: {img_path}, 错误: {e}")
            return None
        return (tensor, img_path)

def collate_fn_remove_corrupted(batch):
    """
    这是一个用于在数据加载器中删除损坏示例的集合函数。
    它期望数据加载器在出现损坏示例时返回None。
    该函数会将批次中的所有None值删除。
    换句话说，当使用此集合函数时，如果数据加载器返回的批次中存在None值，那么它们将被从批次中删除。
    这样可以确保在模型训练或评估过程中不使用损坏的示例数据。
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main(args):
    #加载模型
    train_data_dir = args.train_data_dir
    wdmodel_dir = args.model_dir
    from tensorflow.keras.models import load_model
    model = load_model(f"{wdmodel_dir}")
    #加载csv
    with open(os.path.join(wdmodel_dir, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        header = l[0]  # tag_id,name,category,count
        rows = l[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"意外的CSV格式: {header}"

    general_tags = [row[1] for row in rows[1:] if row[2] == "0"]
    character_tags = [row[1] for row in rows[1:] if row[2] == "4"]

    image_files = [f for f in os.listdir(train_data_dir) if f.endswith(tuple(IMAGE_EXTENSIONS))]
    image_files.sort(key=lambda f: os.path.getsize(os.path.join(train_data_dir, f)))

    """
    for i, file_name in enumerate(image_files, start=1):
        file_extension = os.path.splitext(file_name)[1]
        new_file_name = str(i) + file_extension
        os.rename(os.path.join(train_data_dir, file_name), os.path.join(train_data_dir, new_file_name))
    """

    image_paths = glob_images_pathlib(train_data_dir)
    print(f"共计 {len(image_paths)}张 图片.")

    tag_freq = {}

    undesired_tags = set(args.undesired_tags.split(","))

    def run_batch(path_imgs, output_path=None):
        imgs = np.array([im for _, im in path_imgs])

        probs = model(imgs, training=False)
        probs = probs.numpy()

        for (image_path, _), prob in zip(path_imgs, probs):
            # 前4个标签实际上是评分：选择概率最大的一个
            # ratings_names = label_names[:4]
            # rating_index = ratings_names["probs"].argmax()
            # found_rating = ratings_names[rating_index: rating_index + 1][["name", "probs"]]

            # 其余都是标签：选择预测置信度高于阈值的标签

            combined_tags = []
            general_tag_text = ""
            character_tag_text = ""
            for i, p in enumerate(prob[4:]):
                if i < len(general_tags) and p >= args.general_threshold:
                    tag_name = general_tags[i]
                    if args.remove_underscore and len(tag_name) > 3:  # ignore emoji tags like >_< and ^_^
                        tag_name = tag_name.replace("_", " ")

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        general_tag_text += ", " + tag_name
                        combined_tags.append(tag_name)
                elif i >= len(general_tags) and p >= args.character_threshold:
                    tag_name = character_tags[i - len(general_tags)]
                    if args.remove_underscore and len(tag_name) > 3:
                        tag_name = tag_name.replace("_", " ")

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        character_tag_text += ", " + tag_name
                        combined_tags.append(tag_name)

            # 去掉开头的逗号
            if len(general_tag_text) > 0:
                general_tag_text = general_tag_text[2:]
            if len(character_tag_text) > 0:
                character_tag_text = character_tag_text[2:]

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            if output_path is None:
                caption_file = os.path.splitext(image_path)[0] + args.caption_extension
            else:
                caption_file = os.path.join(output_path, f"{image_name}{args.caption_extension}")

            tag_text = ", ".join(combined_tags)

            if args.append_tags:
                # 检查文件是否存在
                if os.path.exists(caption_file):
                    with open(caption_file, "rt", encoding="utf-8") as f:
                        # 读取文件并删除换行符
                        existing_content = f.read().strip("\n")  # 删除换行符

                    # 将内容分割为标签并存储在列表中
                    existing_tags = [tag.strip() for tag in existing_content.split(",") if tag.strip()]

                    # 检查并删除重复的标签在tag_text中
                    new_tags = [tag for tag in combined_tags if tag not in existing_tags]

                    # 创建新的tag_text
                    tag_text = ", ".join(existing_tags + new_tags)

            with open(caption_file, "wt", encoding="utf-8") as f:
                f.write(tag_text + "\n")


    # 要提高数据加载的速度，可以使用 DataLoader 类来加载和预处理数据
    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingPrepDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is not None:
                image = image.detach().numpy()
            else:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = preprocess_image(image)
                except Exception as e:
                    print(f"无法加载: {image_path}, 错误: {e}")
                    continue
            b_imgs.append((image_path, image))

            if len(b_imgs) >= args.batch_size:
                b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                run_batch(b_imgs,args.output_path)
                b_imgs.clear()

    if len(b_imgs) > 0:
        b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
        run_batch(b_imgs,args.output_path)

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("\n标签频率:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    print("结束!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default='/train', help="用于训练图像的目录")
    parser.add_argument("--output_path", type=str, default=None, help="用于训练图像的目录")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/gemini/pretrain",
        help="存储wd14标签器模型的目录"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="推断时的批处理大小")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=4,
        help="通过DataLoader启用具有此数量工作进程的图像读取（更快）"
    )
    parser.add_argument(
        "--caption_extension", type=str, default=".txt", help="标题文件的扩展名"
    )
    parser.add_argument("--thresh", type=float, default=0.68, help="添加标签的置信度阈值")
    parser.add_argument(
        "--general_threshold",
        type=float,
        default=None,
        help="添加通用类别标签的置信度阈值，如果省略则与--thresh相同"
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=None,
        help="添加字符类别标签的置信度阈值，如果省略则与--thresh相同"
    )
    parser.add_argument(
        "--remove_underscore",
        action="store_true",
        help="在输出标签中用空格替换下划线"
    )
    parser.add_argument(
        "--undesired_tags",
        type=str,
        default="",
        help="要从输出中删除的不希望的标签的逗号分隔列表"
    )
    parser.add_argument("--frequency_tags", action="store_true", help="显示图像标签的标签频率")
    parser.add_argument("--append_tags", action="store_true", help="追加标签而不是覆盖")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh

    main(args)
