from api import Detector
from PIL import Image
import os

# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt',
                    use_cuda=False)

input_dir = "./pic/pic"     # 你的输入图片文件夹
output_dir = "./outputs"   # 输出文件夹
os.makedirs(output_dir, exist_ok=True)

exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

for name in os.listdir(input_dir):
    if not name.lower().endswith(exts):
        continue

    in_path = os.path.join(input_dir, name)

    # 推理 + 返回画好框的 numpy 图片
    np_img = detector.detect_one(
        img_path=in_path,
        input_size=1024,
        conf_thres=0.3,
        return_img=True
    )

    # 输出文件名：原名 + _pred
    base, ext = os.path.splitext(name)
    out_path = os.path.join(output_dir, f"{base}_pred{ext}")

    Image.fromarray(np_img).save(out_path)
    print("Saved:", out_path)









