import numpy as np
import sys
from pathlib import Path
import cv2

# ===== COLOR PICKER =====
# Customize your colors here! Format: (R, G, B) where each value is 0-255
def pick_colors():
    return {
        0: (255, 0, 0),        # Red
        1: (0, 255, 0),        # Green
        2: (0, 255, 255),      # Cyan
        3: (255, 192, 203),    # Pink
        4: (255, 0, 255),      # Magenta
        5: (31, 81, 255),      # Neon Blue
        6: (255, 255, 0),      # Yellow
        7: (120, 81, 169),     # Royal Purple
        8: (18, 10, 143),      # Ultramarine Blue
        9: (255, 165, 0)       # Orange
    }

COLORS = {k: np.array(v, dtype=np.uint8) for k, v in pick_colors().items()}
REVERSED_COLORS = {
    0: COLORS[1], 1: COLORS[0],
    2: COLORS[3], 3: COLORS[2],
    4: COLORS[5], 5: COLORS[4],
    6: COLORS[7], 7: COLORS[6],
    8: COLORS[9], 9: COLORS[8]
}

def colorize(img, label, use_dominant, noise_prob, noise_range,
             stroke_prob, stroke_range, stroke_thick, reverse, all_white):
    
    gray = img[:, :, 0]
    
    if use_dominant:
        if all_white:
            color = np.array([255, 255, 255], dtype=np.uint8)
        else:
            colors = REVERSED_COLORS if reverse else COLORS
            color = colors[label]
    else:
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
    
    # Vectorized color application
    ratio = gray / 255.0
    result = (ratio[:, :, np.newaxis] * color).astype(np.uint8)
    
    # Add noise
    if np.random.random() < noise_prob:
        level = np.random.randint(noise_range[0], noise_range[1])
        noise = np.random.normal(0, level, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add strokes
    if np.random.random() < stroke_prob:
        h, w = result.shape[:2]
        num = np.random.randint(stroke_range[0], stroke_range[1])
        for _ in range(num):
            pt1 = (np.random.randint(w), np.random.randint(h))
            pt2 = (np.random.randint(w), np.random.randint(h))
            color = tuple(np.random.randint(0, 256, 3).tolist())
            thick = np.random.randint(stroke_thick[0], stroke_thick[1])
            cv2.line(result, pt1, pt2, color, thick)
    
    return result

def process_file(input_file, output_file, bias, noise_prob, noise_range,
                stroke_prob, stroke_range, stroke_thick, progress, reverse, all_white):
    
    print(f"Processing {input_file.name}...")
    
    data = np.load(input_file)
    images = data['images']
    labels = data['labels']
    
    total = len(images)
    colored = np.empty_like(images, dtype=np.uint8)
    
    for i in range(total):
        use_dominant = np.random.random() < bias
        colored[i] = colorize(images[i], labels[i], use_dominant,
                             noise_prob, noise_range, stroke_prob, 
                             stroke_range, stroke_thick, reverse, all_white)
        
        if (i + 1) % progress == 0:
            print(f"  {i + 1}/{total} done")
    
    np.savez(output_file, images=colored, labels=labels)
    print(f"Saved to {output_file.name}")

def main():
    # Config
    SAVE_METADATA = True             # Generate metadata file
    ALL_WHITE = False               # If True, all dominant colors become white
    REVERSE = False
    BIAS = 0.95
    NOISE_PROB = 0.7
    NOISE_RANGE = (15, 25)
    STROKE_PROB = 0.5
    STROKE_RANGE = (1, 3)
    STROKE_THICK = (1, 2)
    PROGRESS = 1000
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.glob("*.npz"))
    
    suffix = ""
    if "_" in output_dir.name:
        suffix = output_dir.name[output_dir.name.index("_"):]
    
    if suffix:
        print(f"Adding '{suffix}' to filenames\n")
    
    # Generate metadata file
    if SAVE_METADATA:
        colors = REVERSED_COLORS if REVERSE else COLORS
        metadata_file = output_dir / f"metadata{suffix}.txt"
        
        with open(metadata_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("DATASET METADATA\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("COLOR MAPPING:\n")
            if ALL_WHITE:
                f.write(f"  Mode: ALL WHITE (dominant colors overridden)\n\n")
                for digit in range(10):
                    f.write(f"  Digit {digit}: RGB(255, 255, 255)\n")
            else:
                f.write(f"  Mode: {'REVERSED' if REVERSE else 'NORMAL'}\n\n")
                for digit in range(10):
                    rgb = tuple(colors[digit])
                    f.write(f"  Digit {digit}: RGB{rgb}\n")
            
            f.write("\nPARAMETERS:\n")
            f.write(f"  ALL_WHITE: {ALL_WHITE}\n")
            f.write(f"  REVERSE: {REVERSE}\n")
            f.write(f"  BIAS: {BIAS}\n")
            f.write(f"  NOISE_PROB: {NOISE_PROB}\n")
            f.write(f"  NOISE_RANGE: {NOISE_RANGE}\n")
            f.write(f"  STROKE_PROB: {STROKE_PROB}\n")
            f.write(f"  STROKE_RANGE: {STROKE_RANGE}\n")
            f.write(f"  STROKE_THICK: {STROKE_THICK}\n")
            f.write(f"  PROGRESS: {PROGRESS}\n")
        
        print(f"Metadata saved to {metadata_file.name}\n")
    
    for f in files:
        out = output_dir / f"{f.stem}{suffix}{f.suffix}" if suffix else output_dir / f.name
        process_file(f, out, BIAS, NOISE_PROB, NOISE_RANGE, 
                    STROKE_PROB, STROKE_RANGE, STROKE_THICK, PROGRESS, REVERSE, ALL_WHITE)
        print()
    
    print("Done!")
    print(f"\nProcessed {len(files)} file(s)")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

if __name__ == "__main__":
    main()
