import json,sys,logging,os,re,argparse,concurrent.futures,random
from typing import Optional, List, Tuple, Mapping
import numpy as np
from PIL import Image
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from shtool import unzip_zipfile
import functools

os.environ["HF_OFFLINE"] = "1"

def resize(pic: Image.Image, size: int, keep_ratio: float = True) -> Image.Image:
    """按指定要求调整图像的大小"""
    if not keep_ratio:
        target_size = (size, size)
    else:
        min_edge = min(pic.size)
        target_size = (int(pic.size[0] / min_edge * size), int(pic.size[1] / min_edge * size))
    target_size = ((target_size[0] // 4) * 4, (target_size[1] // 4) * 4)
    return pic.resize(target_size, resample=Image.Resampling.BILINEAR)

def to_tensor(pic: Image.Image):
    """张量,调整和归一化"""
    img: np.ndarray = np.array(pic, np.uint8, copy=True)
    img = img.reshape(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.transpose((2, 0, 1))
    return img.astype(np.float32) / 255

def fill_background(pic: Image.Image, background: str = 'white') -> Image.Image:
    """颜色处理"""
    if pic.mode == 'RGB':
        return pic
    if pic.mode != 'RGBA':
        pic = pic.convert('RGBA')
    background = background or 'white'
    result = Image.new('RGBA', pic.size, background)
    result.paste(pic, (0, 0), pic)
    return result.convert('RGB')

def image_to_tensor(pic: Image.Image, size: int = 512, keep_ratio: float = True, background: str = 'white'):
    return to_tensor(resize(fill_background(pic, background), size, keep_ratio))

@functools.lru_cache(maxsize=None)
def open_onnx_model(ckpt: str, provider: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    logging.info(f'Model {ckpt!r} loaded with provider {provider!r}')
    return InferenceSession(ckpt, options, [provider])

def load_classes(onnx_model_path) -> List[str]:
    classes_file = os.path.join(onnx_model_path, 'classes.json')
    with open(classes_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_tags_from_image( pic: Image.Image, onnx_model_path, ml_model, threshold: float = 0.7, size: int = 512, keep_ratio: bool = False):
    real_input = image_to_tensor(pic, size, keep_ratio)
    real_input = real_input.reshape(1, *real_input.shape)

    native_output, = ml_model.run(['output'], {'input': real_input})

    output = (1 / (1 + np.exp(-native_output))).reshape(-1)
    tags = load_classes(onnx_model_path)
    pairs = sorted([(tags[i], ratio) for i, ratio in enumerate(output)], key=lambda x: (-x[1], x[0]))
    del real_input, native_output, output
    return {tag: float(ratio) for tag, ratio in pairs if ratio >= threshold}

def image_to_mldanbooru_tags(filtered_tags, use_spaces: bool, use_escape: bool, include_ranks: bool, score_descend: bool) \
        -> Tuple[str, Mapping[str, float]]:
    text_items = []
    tags_pairs = filtered_tags.items()
    if score_descend:
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
    for tag, score in tags_pairs:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace('_', ' ')
        if use_escape:
            RE_SPECIAL = re.compile(r'([\\()])')
            tag_outformat = re.sub(RE_SPECIAL, r'\\\1', tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{score:.3f})"
        text_items.append(tag_outformat)
    output_text = ', '.join(text_items)

    return output_text

def process_action(image_path, onnx_model_path, onnx_model_name, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend):
    ml_model = open_onnx_model(os.path.join(onnx_model_path, onnx_model_name), "CUDAExecutionProvider")
    input_image = Image.open(image_path)
    filtered_tags = get_tags_from_image(input_image, onnx_model_path, ml_model, threshold, size, keep_ratio)
    result_text = image_to_mldanbooru_tags(filtered_tags, use_spaces, use_escape, include_ranks, score_descend)
    del image_path, onnx_model_path, onnx_model_name, ml_model, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend, input_image, filtered_tags
    return result_text

def process_image(image_path, onnx_model_path, onnx_model_name, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend, output_path=None, extension="ml_danbooru"):
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_path is None:
        output_file_path = os.path.join(os.path.dirname(image_path), f"{base_name}.{extension}")
    else:
        os.makedirs(output_path, exist_ok=True)
        output_file_path = os.path.join(output_path, f"{base_name}.{extension}")
    result_text = process_action(image_path, onnx_model_path, onnx_model_name, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend)
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(result_text)

"""
def process_images(image_paths, onnx_model_path, model, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend, output_path=None, batch_size=16, extension="ml_danbooru"):
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:  # 使用 max_workers 参数
        futures = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_futures = {executor.submit(process_image, path, onnx_model_path, model, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend): path for path in batch_paths}
            futures.extend(batch_futures)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")
"""

def process_images(image_paths, onnx_model_path, onnx_model_name, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend, output_path=None, extension="ml_danbooru"):
    for path in image_paths:
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_file_path = os.path.join(output_path, f"{base_name}.{extension}")
        if os.path.exists(output_file_path):
            print(f"Skipping {path}. Output file {output_file_path} already exists.")
            continue
        process_image(path, onnx_model_path, onnx_model_name, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend, output_path, extension)

def get_image_paths(temporarily_dir, train_data_dir):
    if os.path.isdir(train_data_dir):
        zip_files = [file for file in os.listdir(train_data_dir) if file.endswith('.zip')]
        if len(zip_files) == 1:
            zip_file_path = os.path.join(train_data_dir, zip_files[0])
        else:
            print("zip文件必须有且仅能有一个")
            exit()
    if train_data_dir.endswith('.zip'):
        zip_file_path = train_data_dir
    os.makedirs(temporarily_dir, exist_ok=True)
    unzip_zipfile(zip_file_path, temporarily_dir, password="shiertier")
    base_name = os.path.splitext(os.path.basename(zip_file_path))[0]
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
    pic_dir = os.path.join(temporarily_dir, base_name)
    image_files = [f for f in os.listdir(pic_dir) if f.endswith(tuple(IMAGE_EXTENSIONS))]
    image_paths = [os.path.join(pic_dir, f) for f in os.listdir(pic_dir) if f.endswith(tuple(IMAGE_EXTENSIONS))]
    random.shuffle(image_paths)
    return image_paths


def main(args):
    temporarily_dir = args.temporarily_dir
    train_data_dir = args.train_data_dir
    pic_dir = args.pic_path
    onnx_model_path = args.onnx_model_path
    onnx_model_name = args.onnx_model_name
    output_path = args.output_path
    #batch_size = args.batch_size
    extension = args.extension
    threshold = args.threshold
    size = args.size
    keep_ratio = args.keep_ratio
    use_spaces = args.use_spaces
    use_escape = args.use_escape
    include_ranks = args.include_ranks
    score_descend = args.score_descend

    if onnx_model_path is None:
        print("请先下载仓库https://huggingface.co/deepghs/ml-danbooru-onnx，并指定存储目录为onnx_model_path")
        sys.exit(1)

    if train_data_dir is not None and pic_dir is not None:
        print("错误：不能同时指定 --train_data_dir 和 --pic_dir。")
        parser.print_help()
        sys.exit(1)

    if train_data_dir is None and pic_dir is None:
        print("错误：--train_data_dir 和 --pic_dir 至少需要指定一个。")
        parser.print_help()
        sys.exit(1)

    if train_data_dir is not None:
        if temporarily_dir is None:
            temporarily_dir = os.path.join(os.path.expanduser("~"), "tarin")
        os.makedirs(temporarily_dir, exist_ok=True)
        image_paths = get_image_paths(temporarily_dir, train_data_dir)
        #process_images(image_paths, onnx_model_path, ml_model, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend, output_path, batch_size, extension)
        process_images(image_paths, onnx_model_path, onnx_model_name, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend, output_path, extension)

    if pic_dir is not None:
        process_image(pic_dir, onnx_model_path, onnx_model_name, threshold, size, keep_ratio, use_spaces, use_escape, include_ranks, score_descend, output_path, extension)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="程序描述")
    parser.add_argument("--temporarily_dir", default=None, help="指定临时目录，当使用压缩包时，图片将解压到这里")
    parser.add_argument("--train_data_dir", default=None, help="指定训练数据目录或目录内的zip文件")
    parser.add_argument("--pic_path", default=None, help="指定单张图片路径")
    parser.add_argument("--onnx_model_path", default=None, help="指定ONNX模型路径")
    parser.add_argument("--onnx_model_name", default="ml_caformer_m36_dec-5-97527.onnx", help="指定ONNX模型名称")
    parser.add_argument("--output_path", default=None, help="指定输出路径")
    #parser.add_argument("--batch_size", type=int, default=1, help="指定批处理大小")
    parser.add_argument("--extension", default="ml_danbooru", help="指定文件扩展名")
    parser.add_argument("--threshold", type=float, default=0.68, help="指定阈值")
    parser.add_argument("--size", type=int, default=960, help="指定图片大小")
    parser.add_argument("--keep_ratio", default=True, action="store_true", help="保持图片比例")
    parser.add_argument("--use_spaces", default=True, action="store_true", help="使用空格替代下划线")
    parser.add_argument("--use_escape", default=False, action="store_true", help="在标签中转义斜杠和括号")
    parser.add_argument("--include_ranks", default=False, action="store_true", help="在输出文本中包含排名")
    parser.add_argument("--score_descend", default=True, action="store_true", help="按分数降序排列标签")

    args = parser.parse_args()
    main(parser.parse_args())

#python ml.py --temporarily_dir "/gemini/code/image" --train_data_dir "/gemini/data-1" --onnx_model_path "/models/ML-Danbooru" --output_path "/gemini/code/output"
#python ml.py --temporarily_dir "/gemini/code/image" --pic_path "/gemini/code/image/aka/danbooru_4024343.jpg" --onnx_model_path "/models/ML-Danbooru" --output_path "/gemini/code/output"
