import os
import sys
from rknn.api import RKNN
import subprocess


def get_first_adb_device():
    # 执行 adb devices 命令并捕获输出
    result = subprocess.run(
        ["adb", "devices"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    output = result.stdout

    # 按照换行符拆分输出结果
    lines = output.strip().split("\n")

    # 除去第一行（标题行），遍历后续行以查找第一个设备
    for line in lines[1:]:
        if line.strip():  # 确保不是空行
            parts = line.split()
            # 如果存在有效的行，并且至少有两部分（确保状态也存在）
            # 第二个元素通常是'device'（如果设备连接正常的话）
            if len(parts) >= 2 and parts[1] == "device":
                return parts[0]  # 返回设备序列号

    # 如果没有找到设备，返回一个空字符串或者None
    return None


def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform]".format(sys.argv[0]))
        print("platform choose from [rk3562,rk3566,rk3568,rk3588]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    return model_path, platform


if __name__ == "__main__":
    model_path, platform = parse_arg()
    DATASET_PATH = f'./{model_path.rsplit(".", 1)[0]}/correction.txt'
    DEFAULT_QUANT = True
    output_path = model_path.rsplit(".", 1)[0] + ".rknn"

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config (quantized_algorithm: kl_divergence, normal, mmse)
    # 量化方式对识别效果影响较大，但并非越慢的量化方式识别效果越好
    # 官方手册说MMSE量化速度较慢，这不正确，应该是非常慢
    print("--> Config model")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform,
        quantized_dtype="asymmetric_quantized-8",
        quantized_algorithm="normal",
        quantized_method="channel",
        optimization_level=3,
        single_core_mode=False,
    )
    print("done")

    # Load model
    print("--> Loading model")
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    ret = rknn.build(do_quantization=DEFAULT_QUANT, dataset=DATASET_PATH)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export rknn model
    print("--> Export rknn model")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")

    device_serial = get_first_adb_device()
    print(f"The first device's serial number is: {device_serial}")
    print("--> Accuracy analysis")
    subset_path = f'{DATASET_PATH.rsplit("/", 1)[0]}/images'
    if device_serial:
        ret = rknn.accuracy_analysis(
            inputs=[f"{subset_path}/{os.listdir(subset_path)[0]}"],
            target=platform,
            device_id=device_serial,
        )
    else:
        ret = rknn.accuracy_analysis(
            inputs=[f"{subset_path}/{os.listdir(subset_path)[0]}"],
            target=None,
        )

    # Release
    rknn.release()
