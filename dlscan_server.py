from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Tuple
from typing_extensions import TypedDict
import sys
import numpy as np
import tifffile
from skimage import exposure

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent

mcp = FastMCP("dlscan")

# this is needed because it is efficient to give the tool input as .npy rather than an array to avoid overload
class ArrayHandle(TypedDict):
    path: str
    shape: Tuple[int, ...]
    dtype: str


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_npy(arr: np.ndarray, out_path: Path) -> ArrayHandle:
    _ensure_dir(out_path.parent)
    np.save(out_path, arr)
    return {"path": str(out_path), "shape": tuple(arr.shape), "dtype": str(arr.dtype)}


@mcp.tool()
def ingest_tiff(
    tiff_path: str,
    work_dir: str,
    out_dir: str,
    name_raw: str = "stack_raw",
    name_uint8: str = "stack_uint8",
    per_frame: bool = True,
    save: bool = True,
) -> dict:
    """
    Combine:
      - load_tiff_stack (TIFF -> numpy stack)
      - to_uint8_stack (stack -> uint8)

    If save=True:
      - saves raw and uint8 stacks as .npy and returns ArrayHandles with real paths (needed because the pipeline takes input from the previous output)
    If save=False:
      - does NOT save; returns ArrayHandles with path="" but correct shape/dtype (avoid; files can manually be delted later)

    Returns:
      {"raw": ArrayHandle, "uint8": ArrayHandle}
    """
    p = Path(tiff_path)
    if not p.exists():
        raise FileNotFoundError(f"TIFF not found: {p}")

    arr = tifffile.imread(str(p))

    # dlscan assumes a stack. if it's 2D, it's a single image
    if arr.ndim == 2:
        raise ValueError(f"Expected TIFF stack, got single 2D image shape={arr.shape}")

    # convert to uint8 (same logic as to_uint8_stack) 
    if arr.dtype == np.uint8:
        out = arr
    else:
        if arr.ndim not in (3, 4):
            raise ValueError(f"Unsupported stack ndim={arr.ndim}, shape={arr.shape}")

        if per_frame:
            out = np.empty(arr.shape, dtype=np.uint8)
            for t in range(arr.shape[0]):
                out[t] = exposure.rescale_intensity(arr[t], out_range=(0, 255)).astype(np.uint8)
        else:
            out = exposure.rescale_intensity(arr, out_range=(0, 255)).astype(np.uint8)

    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()

        # enforce saving inside project folder
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"work_dir must be inside project folder. Got: {output_dir}")
        _ensure_dir(output_dir)

        raw_handle = _save_npy(arr, output_dir / f"{name_raw}.npy")
        u8_handle = _save_npy(out, output_dir / f"{name_uint8}.npy")

    else:
        raw_handle = {"path": "", "shape": tuple(arr.shape), "dtype": str(arr.dtype)}
        u8_handle = {"path": "", "shape": tuple(out.shape), "dtype": str(out.dtype)}

    return {"raw": raw_handle, "uint8": u8_handle}


@mcp.tool()
def preprocess_gaussian_blur_stack(
    stack_npy_path: str,
    ksize: int,
    work_dir: str,
    out_dir: str,
    name_out: str = "stack_gaussian",
    save: bool = True,
) -> ArrayHandle:
    """
    Apply Gaussian blur to each frame of a stack.

    dlsacn convention:
      ksize = -1  → skip entirely (identity)
      otherwise: odd integer > 0 (e.g. 1,3,5,...)
    """
    import numpy as np
    import cv2
    from skimage import exposure
    from pathlib import Path

    # load
    p = Path(stack_npy_path)
    if not p.exists():
        raise FileNotFoundError(f".npy not found: {p}")

    stack = np.load(str(p), allow_pickle=False)

    #ensure grayscale 
    if stack.ndim == 4 and stack.shape[-1] == 3:
        if stack.dtype != np.uint8:
            stack = exposure.rescale_intensity(stack, out_range=(0, 255)).astype(np.uint8)
        gray = np.empty(stack.shape[:3], dtype=np.uint8)
        for t in range(stack.shape[0]):
            gray[t] = cv2.cvtColor(stack[t], cv2.COLOR_RGB2GRAY)
        stack = gray

    if stack.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got shape={stack.shape}")

    # ensure uint8
    if stack.dtype != np.uint8:
        stack_u8 = np.empty(stack.shape, dtype=np.uint8)
        for t in range(stack.shape[0]):
            stack_u8[t] = exposure.rescale_intensity(stack[t], out_range=(0, 255)).astype(np.uint8)
    else:
        stack_u8 = stack

    # gaussian blur (DLSCAN skip on -1) 
    if ksize == -1:
        out = stack_u8  # skip entirely: no opencv call
    else:
        if ksize <= 0 or ksize % 2 == 0:
            raise ValueError("ksize must be -1 or an odd integer")

        out = stack_u8.copy()
        for t in range(out.shape[0]):
            out[t] = cv2.GaussianBlur(out[t], (ksize, ksize), 0)

    #save 
    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"out_dir must be inside project directory. Got: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{name_out}.npy"
        np.save(out_path, out)
        return {"path": str(out_path), "shape": tuple(out.shape), "dtype": str(out.dtype)}

    return {"path": "", "shape": tuple(out.shape), "dtype": str(out.dtype)}



@mcp.tool()
def preprocess_median_blur_stack(
    stack_npy_path: str,
    ksize: int,
    work_dir: str,
    out_dir: str,
    name_out: str = "stack_median",
    save: bool = True,
) -> ArrayHandle:
    """
    Apply Median blur to each frame of a stack.

    DLScan convention:
      ksize = -1  → skip entirely (identity)
      otherwise: odd integer >= 3
    """
    import numpy as np
    import cv2
    from skimage import exposure
    from pathlib import Path

    #load 
    p = Path(stack_npy_path)
    if not p.exists():
        raise FileNotFoundError(f".npy not found: {p}")

    stack = np.load(str(p), allow_pickle=False)

    #ensure grayscale
    if stack.ndim == 4 and stack.shape[-1] == 3:
        if stack.dtype != np.uint8:
            stack = exposure.rescale_intensity(stack, out_range=(0, 255)).astype(np.uint8)
        gray = np.empty(stack.shape[:3], dtype=np.uint8)
        for t in range(stack.shape[0]):
            gray[t] = cv2.cvtColor(stack[t], cv2.COLOR_RGB2GRAY)
        stack = gray

    if stack.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got shape={stack.shape}")

    # ensure uint8 
    if stack.dtype != np.uint8:
        stack_u8 = np.empty(stack.shape, dtype=np.uint8)
        for t in range(stack.shape[0]):
            stack_u8[t] = exposure.rescale_intensity(stack[t], out_range=(0, 255)).astype(np.uint8)
    else:
        stack_u8 = stack

    #median blur (DLSCAN skip on -1) 
    if ksize == -1:
        out = stack_u8  # skip entirely
    else:
        if ksize < 3 or ksize % 2 == 0:
            raise ValueError("ksize must be -1 or an odd integer >= 3")

        out = stack_u8.copy()
        for t in range(out.shape[0]):
            out[t] = cv2.medianBlur(out[t], ksize)

    # save 
    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"out_dir must be inside project directory. Got: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{name_out}.npy"
        np.save(out_path, out)
        return {"path": str(out_path), "shape": tuple(out.shape), "dtype": str(out.dtype)}

    return {"path": "", "shape": tuple(out.shape), "dtype": str(out.dtype)}


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    import numpy as np
    import cv2
    """
    EXACT DL-SCAN implementation.
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness

        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        buf = np.clip(buf, 0, 255).astype(np.uint8)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        buf = np.clip(buf, 0, 255).astype(np.uint8)

    return buf


@mcp.tool()
def preprocess_brightness_contrast_stack(
    stack_npy_path: str,
    brightness: int,
    contrast: int,
    work_dir: str,
    out_dir: str,
    name_out: str = "stack_bc",
    save: bool = True,
) -> ArrayHandle:
    """
    DL-SCAN brightness / contrast preprocessing.

    Conventions:
      brightness = -1 and contrast = -1  → skip entirely
      otherwise uses EXACT DLSCAN apply_brightness_contrast()

    Input:
      - (T,H,W) or (T,H,W,3)
    Output:
      - (T,H,W) uint8
    """
    import numpy as np
    import cv2
    from skimage import exposure
    from pathlib import Path

    #load 
    p = Path(stack_npy_path)
    if not p.exists():
        raise FileNotFoundError(f".npy not found: {p}")

    stack = np.load(str(p), allow_pickle=False)

    #ensure grayscale
    if stack.ndim == 4 and stack.shape[-1] == 3:
        if stack.dtype != np.uint8:
            stack = exposure.rescale_intensity(stack, out_range=(0, 255)).astype(np.uint8)

        gray = np.empty(stack.shape[:3], dtype=np.uint8)
        for t in range(stack.shape[0]):
            gray[t] = cv2.cvtColor(stack[t], cv2.COLOR_RGB2GRAY)
        stack = gray

    if stack.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got shape={stack.shape}")

    # ensure uint8
    if stack.dtype != np.uint8:
        stack_u8 = np.empty(stack.shape, dtype=np.uint8)
        for t in range(stack.shape[0]):
            stack_u8[t] = exposure.rescale_intensity(
                stack[t], out_range=(0, 255)
            ).astype(np.uint8)
    else:
        stack_u8 = stack

    #  brightness / contrast (from DL-SCAN)
    if brightness == -1 and contrast == -1:
        out = stack_u8  # skip entirely
    else:
        b = 0 if brightness == -1 else int(brightness)
        c = 0 if contrast == -1 else int(contrast)

        out = stack_u8.copy()
        for t in range(out.shape[0]):
            out[t] = apply_brightness_contrast(out[t], b, c)

    # ave 
    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"out_dir must be inside project directory. Got: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{name_out}.npy"
        np.save(out_path, out)
        return {
            "path": str(out_path),
            "shape": tuple(out.shape),
            "dtype": str(out.dtype),
        }

    return {"path": "", "shape": tuple(out.shape), "dtype": str(out.dtype)}

@mcp.tool()
def preprocess_clahe_stack(
    stack_npy_path: str,
    clip_limit: float,
    tile_grid_size: int,
    work_dir: str,
    out_dir: str,
    name_out: str = "stack_clahe",
    save: bool = True,
) -> ArrayHandle:
    """
    DL-SCAN CLAHE (Adaptive Histogram Equalization) EXACTLY:

      1) (per frame) RGB -> LAB
      2) split into L, a, b
      3) CLAHE on L only
      4) merge back -> LAB
      5) LAB -> RGB
      6) take channel 0: RGB[:,:,0]  -> grayscale frame

    Convention:
      clip_limit = -1  → skip entirely (identity; no conversion, no rescale)

    Input:
      - (T,H,W) or (T,H,W,3)

    Output when enabled:
      - (T,H,W) uint8  (DL-SCAN uses RGB[:,:,0])

    Output when skipped:
      - identical to input (same shape/dtype)
    """
    import numpy as np
    import cv2
    from skimage import exposure
    from pathlib import Path

    # load
    p = Path(stack_npy_path)
    if not p.exists():
        raise FileNotFoundError(f".npy not found: {p}")

    stack = np.load(str(p), allow_pickle=False)

    # skip entirely 
    if clip_limit == -1:
        out = stack
    else:
        if clip_limit <= 0:
            raise ValueError("clip_limit must be -1 (skip) or > 0")
        if tile_grid_size <= 0:
            raise ValueError("tile_grid_size must be a positive integer")

        if stack.ndim not in (3, 4):
            raise ValueError(f"Expected (T,H,W) or (T,H,W,3), got shape={stack.shape}")

        T = stack.shape[0]

        #ensure uint8 (OpenCV CLAHE expects uint8)
        if stack.dtype != np.uint8:
            # rescale to [0,255] like other DLSCAN preprocessing steps
            stack_u8 = stack.astype(np.uint8) 
        else:
            stack_u8 = stack

        # ensure RGB frames (T,H,W,3)
        if stack_u8.ndim == 3:
            # grayscale ->  RGB by repeating channels
            rgb_stack = np.repeat(stack_u8[:, :, :, None], 3, axis=3)
        else:
            if stack_u8.shape[-1] != 3:
                raise ValueError(f"Expected RGB last dim=3, got shape={stack_u8.shape}")
            rgb_stack = stack_u8

        #DLSCAN CLAHE on LAB L-channel
        out = np.empty(rgb_stack.shape[:3], dtype=np.uint8)  # (T,H,W)
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
        )

        for t in range(T):
            lab = cv2.cvtColor(rgb_stack[t], cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
            out[t] = rgb2[:, :, 0]  # EXACTLY what DLSCAN does

        # (out is already uint8 in [0,255])

    # save 
    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"out_dir must be inside project directory. Got: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{name_out}.npy"
        np.save(out_path, out)
        return {"path": str(out_path), "shape": tuple(out.shape), "dtype": str(out.dtype)}

    return {"path": "", "shape": tuple(out.shape), "dtype": str(out.dtype)}

@mcp.tool()
def preprocess_collapsed_max_image(
    stack_npy_path: str,
    mode: int,
    work_dir: str,
    out_dir: str,
    name_out: str = "collapsed_MIP_or_first",
    save: bool = True,
) -> ArrayHandle:
    """
    Generate DL-SCAN "collapsed image" / "super image".

    Convention:
      mode = -1  → first frame (as a 2D image)
      mode =  1  → max projection over time

    Input:
      - (T,H,W) preferred
      - also accepts (T,H,W,3) and uses channel 0

    Output:
      - (H,W) uint8
    """
    import numpy as np
    from pathlib import Path

    # -------- load --------
    p = Path(stack_npy_path)
    if not p.exists():
        raise FileNotFoundError(f".npy not found: {p}")

    stack = np.load(str(p), allow_pickle=False)

    # pick image
    if mode == -1:
        # FIRST frame
        if stack.ndim == 3:
            img = stack[0]
        elif stack.ndim == 4 and stack.shape[-1] == 3:
            img = stack[0, :, :, 0]
        else:
            raise ValueError(f"Expected (T,H,W) or (T,H,W,3), got shape={stack.shape}")

    elif mode == 1:
        # MIp or collapsed image
        if stack.ndim == 3:
            img = stack.max(axis=0)
        elif stack.ndim == 4 and stack.shape[-1] == 3:
            img = stack.max(axis=0)[:, :, 0]
        else:
            raise ValueError(f"Expected (T,H,W) or (T,H,W,3), got shape={stack.shape}")

    else:
        raise ValueError("mode must be -1 (first frame) or 1 (max projection)")

    # DLSCAN implemnetation
    img_i32 = img.astype(np.int32)
    img_i32[img_i32 > 255] = 255
    img_i32[img_i32 < 0] = 0
    out = img_i32.astype(np.uint8)

    #save 
    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"out_dir must be inside project directory. Got: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{name_out}.npy"
        np.save(out_path, out)
        return {"path": str(out_path), "shape": tuple(out.shape), "dtype": str(out.dtype)}

    return {"path": "", "shape": tuple(out.shape), "dtype": str(out.dtype)}

@mcp.tool()
def preprocess_rolling_ball_bg_sub_image(
    image_npy_path: str,
    radius: int,
    work_dir: str,
    out_dir: str,
    name_out: str = "image_rball_bgsub",
    save: bool = True,
) -> ArrayHandle:
    """
    Rolling-ball background subtraction on a SINGLE 2D image (DL-SCAN-style, segmentation aid).

    Convention:
      radius = -1  → identity (no-op; no conversion/rescale)
      radius > 0   → apply rolling ball background correction:
                    out = clip(image - background, 0..255).astype(uint8)

    Input:
      - (H,W) image .npy (typically the collapsed image / super_im)

    Output:
      - (H,W) uint8 when enabled
      - unchanged shape/dtype when radius == -1
    """
    import numpy as np
    from pathlib import Path
    from skimage import exposure, restoration

    #load 
    p = Path(image_npy_path)
    if not p.exists():
        raise FileNotFoundError(f".npy not found: {p}")

    img = np.load(str(p), allow_pickle=False)

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image (H,W), got shape={img.shape}")

    # no RBBC
    if radius == -1:
        out = img
    else:
        if radius <= 0:
            raise ValueError("radius must be -1 (skip) or an integer > 0")

        # ensure uint8  (DL-SCAN uses uint8)
        if img.dtype != np.uint8:
            img_u8 = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
        else:
            img_u8 = img

        # rolling ball background estimate
        bg = restoration.rolling_ball(img_u8, radius=radius)

        # subtraction
        out_i32 = img_u8.astype(np.int32) - np.round(bg).astype(np.int32)
        out = np.clip(out_i32, 0, 255).astype(np.uint8)

    # save
    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"out_dir must be inside project directory. Got: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{name_out}.npy"
        np.save(out_path, out)
        return {"path": str(out_path), "shape": tuple(out.shape), "dtype": str(out.dtype)}

    return {"path": "", "shape": tuple(out.shape), "dtype": str(out.dtype)}


def postprocess_labels(label):
    """
    DL-SCAN-style postprocess:
      - regionprops to compute equivalent_diameter_area
      - keep labels with diameter > 0.5 * mean(diameters)
      - relabel objects sorted by centroid (row then col)
    Returns: final_label uint8
    """
    import numpy as np
    import statistics as stat
    from skimage import measure

    props = measure.regionprops(label)
    if len(props) == 0:
        return np.zeros_like(label, dtype=np.uint8)

    diam = [props[i]["equivalent_diameter_area"] for i in range(len(props))]
    mean_d = stat.mean(diam)

    #dlscan uses the count of objects passing the filter, then assumes labels 1..N exist
    labels_to_keep_len = len([p["label"] for p in props if p["equivalent_diameter_area"] > 0.5 * mean_d])
    labels_to_keep = list(range(1, labels_to_keep_len + 1))

    label_f = np.zeros_like(label, dtype=np.uint8)
    for lab in labels_to_keep:
        label_f[label == lab] = lab

    props2 = measure.regionprops(label_f)
    if len(props2) == 0:
        return np.zeros_like(label_f, dtype=np.uint8)

    centroids = np.array([p.centroid for p in props2])  # (row, col)
    sorted_indices = np.lexsort((centroids[:, 1], centroids[:, 0]))  # sort by row then col

    final_label = np.zeros_like(label_f, dtype=np.uint8)
    for new_id, idx in enumerate(sorted_indices, start=1):
        # repo uses (idx+1) when mapping back (because labels in label_f are 1..N)
        final_label[label_f == (idx + 1)] = new_id

    return final_label


def postprocess_labels_rgb(final_label_u8):
    """
    DL-SCAN-style visualization:
      - black background
      - each object filled red
      - object ID text in white near minEnclosingCircle center
    Returns: (H,W,3) uint8
    """
    import numpy as np
    import cv2
    import imutils
    from skimage.util import img_as_ubyte

    if final_label_u8.ndim != 2:
        raise ValueError(f"final_label must be 2D, got shape={final_label_u8.shape}")

    labels_rgb = np.expand_dims(final_label_u8, axis=2)
    final_label_rgb = cv2.cvtColor(img_as_ubyte(labels_rgb), cv2.COLOR_GRAY2RGB)

    for lab in range(1, int(final_label_u8.max()) + 1):
        mask = np.zeros(final_label_u8.shape, dtype="uint8")
        mask[final_label_u8 == lab] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)

        ((x, y), r) = cv2.minEnclosingCircle(c)

        # fill the object red
        final_label_rgb[final_label_u8 == lab] = (255, 0, 0)

        # write label in white
        cv2.putText(
            final_label_rgb,
            "{}".format(lab),
            (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )

    return final_label_rgb


#cache the model to prevent reloadng everytime
@lru_cache(maxsize=4)
def _get_stardist_model(model_name: str):
    from stardist.models import StarDist2D
    return StarDist2D.from_pretrained(model_name)

@mcp.tool()
def segment_stardist_2d(
    image_npy_path: str,
    out_dir: str,
    work_dir: str,
    name_out: str = "stardist",
    model_name: str = "2D_versatile_fluo",
    prob_thresh: float = 0.6,
    save: bool = True,
) -> dict:
    """
    DL-SCAN-faithful StarDist segmentation:
      1) StarDist predict_instances(normalize(img), prob_thresh)
      2) regionprops filter + centroid-sorted relabel -> final_label
      3) generate final_label_rgb (black bg, red objects, white IDs)
      4) save outputs
    """
    import numpy as np
    from pathlib import Path
    from csbdeep.utils import normalize
    from imageio.v2 import imwrite

    p = Path(image_npy_path)
    if not p.exists():
        raise FileNotFoundError(f".npy not found: {p}")

    img = np.load(str(p), allow_pickle=False)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image (H,W), got shape={img.shape}")

    model = _get_stardist_model(model_name)

    raw_labels, _ = model.predict_instances(normalize(img), prob_thresh=float(prob_thresh))
    raw_labels = raw_labels.astype(np.int32, copy=False)

    final_label = postprocess_labels(raw_labels)          # uint8
    final_label_rgb = postprocess_labels_rgb(final_label)            # uint8 RGB

    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"out_dir must be inside project directory. Got: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_path = output_dir / f"{name_out}_raw_labels.npy"
        final_path = output_dir / f"{name_out}_final_label.npy"
        rgb_path = output_dir / f"{name_out}_final_label_rgb.png"

        np.save(raw_path, raw_labels)
        np.save(final_path, final_label)
        imwrite(rgb_path, final_label_rgb)

        return {
            "raw_labels": {"path": str(raw_path), "shape": tuple(raw_labels.shape), "dtype": str(raw_labels.dtype)},
            "final_label": {"path": str(final_path), "shape": tuple(final_label.shape), "dtype": str(final_label.dtype)},
            "final_label_rgb": {"path": str(rgb_path), "shape": tuple(final_label_rgb.shape), "dtype": str(final_label_rgb.dtype)},
            "model_name": model_name,
            "prob_thresh": float(prob_thresh),
            "n_final": int(final_label.max()),
        }

    return {
        "raw_labels": {"path": "", "shape": tuple(raw_labels.shape), "dtype": str(raw_labels.dtype)},
        "final_label": {"path": "", "shape": tuple(final_label.shape), "dtype": str(final_label.dtype)},
        "final_label_rgb": {"path": "", "shape": tuple(final_label_rgb.shape), "dtype": str(final_label_rgb.dtype)},
        "model_name": model_name,
        "prob_thresh": float(prob_thresh),
        "n_final": int(final_label.max()),
    }

@mcp.tool()
def extract_label_intensity_area_table(
    stack_npy_path: str,
    label_npy_path: str,
    area_thres: float,
    work_dir: str,
    out_dir: str,
    name_out: str = "label_intensity_data",
    channel: int = 0,
    save: bool = True,
) -> dict:
    """
    DL-SCAN-matching table (column order):
      label,
      intensity_mean_0..T-1,
      Bright_pixel_area_0..T-1

    Note:
      We still compute intensity_max internally to compute Bright_pixel_area,
      but we do NOT include intensity_max columns in the CSV.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from skimage import exposure, measure

    if not (0.0 <= float(area_thres) <= 1.0):
        raise ValueError("area_thres must be in [0,1], e.g. 0.30")

    sp = Path(stack_npy_path)
    lp = Path(label_npy_path)
    if not sp.exists():
        raise FileNotFoundError(f"stack .npy not found: {sp}")
    if not lp.exists():
        raise FileNotFoundError(f"label .npy not found: {lp}")

    stack = np.load(str(sp), allow_pickle=False)
    labels = np.load(str(lp), allow_pickle=False)

    if labels.ndim != 2:
        raise ValueError(f"Expected label mask (H,W), got shape={labels.shape}")

    # stack can be (T,H,W) or (T,H,W,C)
    if stack.ndim == 4:
        if channel < 0 or channel >= stack.shape[-1]:
            raise ValueError(f"channel out of range. stack.shape[-1]={stack.shape[-1]}, got channel={channel}")
        stack2 = stack[:, :, :, channel]
    elif stack.ndim == 3:
        stack2 = stack
    else:
        raise ValueError(f"Expected stack (T,H,W) or (T,H,W,C), got shape={stack.shape}")

    T, H, W = stack2.shape
    if (H, W) != labels.shape:
        raise ValueError(f"Stack frame shape {(H,W)} must match labels shape {labels.shape}")

    #normalize to uint8 like dlsacn pipeline
    if stack2.dtype != np.uint8:
        stack_u8 = exposure.rescale_intensity(stack2, out_range=(0, 255)).astype(np.uint8)
    else:
        stack_u8 = stack2

    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int32)

    n_labels = int(labels.max())
    if n_labels == 0:
        df = pd.DataFrame({"label": []})
    else:
        mean_mat = np.zeros((n_labels, T), dtype=np.float64)
        max_mat = np.zeros((n_labels, T), dtype=np.float64)

        for t in range(T):
            stats_mean = measure.regionprops_table(
                labels, intensity_image=stack_u8[t], properties=["label", "intensity_mean"]
            )
            stats_max = measure.regionprops_table(
                labels, intensity_image=stack_u8[t], properties=["label", "intensity_max"]
            )

            lab_ids = np.asarray(stats_mean["label"], dtype=np.int32)
            mean_vals = np.asarray(stats_mean["intensity_mean"], dtype=np.float64)
            max_vals = np.asarray(stats_max["intensity_max"], dtype=np.float64)

            mean_mat[lab_ids - 1, t] = mean_vals
            max_mat[lab_ids - 1, t] = max_vals

        # bright pixel area (pixels > area_thres * max intensity per label per frame)
        bright_area = np.zeros((n_labels, T), dtype=np.float64)
        idxs = [None] * (n_labels + 1)
        for lab in range(1, n_labels + 1):
            idxs[lab] = np.where(labels == lab)

        thr = float(area_thres)
        for lab in range(1, n_labels + 1):
            rr, cc = idxs[lab]
            if rr.size == 0:
                continue
            for t in range(T):
                cutoff = thr * max_mat[lab - 1, t]
                bright_area[lab - 1, t] = float(np.sum(stack_u8[t][rr, cc] > cutoff))

        df = pd.DataFrame({"label": np.arange(1, n_labels + 1, dtype=np.int32)})

        for t in range(T):
            df[f"intensity_mean_{t}"] = np.round(mean_mat[:, t], 3)

        for t in range(T):
            df[f"Bright_pixel_area_{t}"] = bright_area[:, t].astype(np.float64)

    if save:
        output_dir = (PROJECT_ROOT / out_dir).resolve()
        if PROJECT_ROOT not in output_dir.parents and output_dir != PROJECT_ROOT:
            raise ValueError(f"out_dir must be inside project directory. Got: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{name_out}.csv"
        df.to_csv(csv_path, index=False)
        return {"csv_path": str(csv_path), "n_labels": int(n_labels), "n_frames": int(T)}

    return {"csv_path": "", "n_labels": int(n_labels), "n_frames": int(T)}


def pre_warm_models():
    """to avoid timeout later, force StarDist to load into RAM during server startup."""
    #stderr so it shows up in claude's logs
    print("Pre-warming StarDist model... please wait.", file=sys.stderr)
    try:
        #calls your existing cached function with the default model
        _get_stardist_model('2D_versatile_fluo')
        print("StarDist model pre-loaded successfully!", file=sys.stderr)
    except Exception as e:
        print(f"Pre-warm failed (this is okay, tool will load it later): {e}", file=sys.stderr)

if __name__ == "__main__":
    #load the model FIRST while Claude is still initializing the server to bypass timeout
    pre_warm_models()
    #tehn run the server
    mcp.run()
