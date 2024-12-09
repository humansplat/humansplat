import argparse
import glob
import logging
import os

import cv2
import numpy as np
import rembg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Remove background and recenter the image of an object"
    )

    parser.add_argument("path", type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument(
        "--model",
        default="u2net",
        type=str,
        help="rembg model, see https://github.com/danielgatis/rembg#models",
    )
    parser.add_argument("--size", default=512, type=int, help="output resolution")
    parser.add_argument(
        "--border_ratio", default=0.05, type=float, help="output border ratio"
    )
    parser.add_argument(
        "--recenter",
        type=bool,
        default=True,
        help="Recenter the object, potentially not helpful for multiview zero123",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - REMBG&CENTER - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.propagate = True  # propagate to the root logger (console)

    # Create a session for rembg
    session = rembg.new_session(model_name=args.model)

    if os.path.isdir(args.path):
        logger.info(f"Processing directory [{args.path}]")
        files = glob.glob(f"{args.path}/*")
        out_dir = args.path
    else:  # single file
        files = [args.path]
        out_dir = os.path.dirname(args.path)

    for file in files:
        out_base = os.path.basename(file).split(".")[0]
        out_rgba = os.path.join(out_dir, out_base + "_rgba.png")

        # Load image and resize
        logger.info(f"Load image [{file}]")
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        _h, _w = image.shape[:2]
        scale = args.size / max(_h, _w)
        _h, _w = int(_h * scale), int(_w * scale)
        image = cv2.resize(image, (_w, _h), interpolation=cv2.INTER_AREA)

        # Remove background
        logger.info("Removing background")
        carved_image = rembg.remove(image, session=session)  # (H, W, 4)
        mask = carved_image[..., -1] > 0

        # Recenter the object
        if args.recenter:
            logger.info("Recentering object")
            final_rgba = np.zeros((args.size, args.size, 4), dtype=np.uint8)

            coords = np.nonzero(mask)
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            h = x_max - x_min
            w = y_max - y_min
            desired_size = int(args.size * (1 - args.border_ratio))
            scale = desired_size / max(h, w)
            h2 = int(h * scale)
            w2 = int(w * scale)
            x2_min = (args.size - h2) // 2
            x2_max = x2_min + h2
            y2_min = (args.size - w2) // 2
            y2_max = y2_min + w2
            final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
                carved_image[x_min:x_max, y_min:y_max],
                (w2, h2),
                interpolation=cv2.INTER_AREA,
            )
        else:
            final_rgba = carved_image

        # Save image
        cv2.imwrite(out_rgba, final_rgba)

    print()  # newline after the process
