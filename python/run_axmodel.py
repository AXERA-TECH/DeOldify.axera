import argparse
import cv2
import numpy as np
import axengine as axe

def from_numpy(x):
    return x if isinstance(x, np.ndarray) else np.array(x)
    
def post_process(raw_color, orig):
    color_np = np.asarray(raw_color)
    orig_np = np.asarray(orig)
    color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)
    # do a black and white transform first to get better luminance values
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2YUV)
    hires = np.copy(orig_yuv)
    hires[:, :, 1:3] = color_yuv[:, :, 1:3]
    final = cv2.cvtColor(hires, cv2.COLOR_YUV2RGB)
    return final

def main(args):
    # Initialize the model
    session = axe.InferenceSession(args.model_path)
    output_names = [x.name for x in session.get_outputs()]
    input_name = session.get_inputs()[0].name
    print(input_name)
    print(output_names)

    ori_image = cv2.imread(args.input_path)
    h, w = ori_image.shape[:2]
    image = cv2.resize(ori_image, (512, 512))
    image = (image[..., ::-1] /255.0).astype(np.float32)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = ((image - mean) / std).astype(np.float32)

    #image = (image /1.0).astype(np.float32)
    image = np.transpose(np.expand_dims(np.ascontiguousarray(image), axis=0), (0,3,1,2))
    print(image.shape)

    
    # Use the model to generate super-resolved images
    sr = session.run(output_names, {input_name: image})
    
    if isinstance(sr, (list, tuple)):
        sr = from_numpy(sr[0]) if len(sr) == 1 else [from_numpy(x) for x in sr]
    else:
        sr = from_numpy(sr)

    #sr_y_image = imgproc.array_to_image(sr)
    sr = np.transpose(sr.squeeze(0), (1,2,0))
    sr = (sr*std + mean).astype(np.float32)
    
    # Save image
    ndarr = np.clip((sr*255.0), 0, 255.0).astype(np.uint8)
    ndarr = cv2.resize(ndarr[..., ::-1], (w, h))
    out_image = post_process(ndarr, ori_image)

    cv2.imwrite(args.output_path, out_image)
    print(f"Color image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--input_path",
                        type=str,
                        default="./input.png",
                        help="origin image path.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./sr_colorized.jpg",
                        help="colorized image path.")
    parser.add_argument("--model_path",
                        type=str,
                        default="./colorize_stable.axmodel",
                        help="model path.")
    args = parser.parse_args()

    main(args)
