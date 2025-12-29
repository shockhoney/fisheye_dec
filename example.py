from api import Detector
from PIL import Image
import os

# Initialize detector
detector = Detector(model_name='rapid',
                            weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt',
                                                use_cuda=False)

# 让 detect_one 返回“画好框”的 numpy 图片 :contentReference[oaicite:2]{index=2}
np_img = detector.detect_one(
            img_path='./images/1.jpg',
                input_size=1024, conf_thres=0.3,
                    return_img=True
                    )

# 保存到 outputs/
os.makedirs("outputs", exist_ok=True)
Image.fromarray(np_img).save("outputs/exhibition_pred.jpg")
print("Saved: outputs/exhibition_pred.jpg")








