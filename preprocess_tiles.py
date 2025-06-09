import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import json


def load_yolo_annotations(annotation_file, img_width, img_height):
    """Load YOLO format annotations and convert to absolute coordinates"""
    annotations = []

    if not os.path.exists(annotation_file) or os.path.getsize(annotation_file) == 0:
        return annotations

    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    category = int(parts[0])
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height

                    # Convert to bounding box (x1, y1, x2, y2)
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2

                    annotations.append({
                        'category': category,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'center_x': center_x, 'center_y': center_y,
                        'width': width, 'height': height
                    })

    return annotations


def get_tile_annotations(annotations, tile_x, tile_y, tile_width, tile_height, min_area_ratio=0.3):
    """Get annotations that overlap with a tile and convert to tile coordinates"""
    tile_annotations = []

    tile_x1, tile_y1 = tile_x, tile_y
    tile_x2, tile_y2 = tile_x + tile_width, tile_y + tile_height

    for ann in annotations:
        # Check if annotation overlaps with tile
        if (ann['x2'] > tile_x1 and ann['x1'] < tile_x2 and
                ann['y2'] > tile_y1 and ann['y1'] < tile_y2):

            # Calculate intersection
            intersect_x1 = max(ann['x1'], tile_x1)
            intersect_y1 = max(ann['y1'], tile_y1)
            intersect_x2 = min(ann['x2'], tile_x2)
            intersect_y2 = min(ann['y2'], tile_y2)

            # Calculate area ratios
            intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
            original_area = ann['width'] * ann['height']
            area_ratio = intersect_area / original_area if original_area > 0 else 0

            # Only include if significant overlap
            if area_ratio >= min_area_ratio:
                # Convert to tile coordinates
                tile_ann_x1 = max(0, ann['x1'] - tile_x1)
                tile_ann_y1 = max(0, ann['y1'] - tile_y1)
                tile_ann_x2 = min(tile_width, ann['x2'] - tile_x1)
                tile_ann_y2 = min(tile_height, ann['y2'] - tile_y1)

                # Convert to YOLO format (normalized center coordinates)
                tile_center_x = (tile_ann_x1 + tile_ann_x2) / 2 / tile_width
                tile_center_y = (tile_ann_y1 + tile_ann_y2) / 2 / tile_height
                tile_norm_width = (tile_ann_x2 - tile_ann_x1) / tile_width
                tile_norm_height = (tile_ann_y2 - tile_ann_y1) / tile_height

                # Ensure values are within [0, 1]
                tile_center_x = max(0, min(1, tile_center_x))
                tile_center_y = max(0, min(1, tile_center_y))
                tile_norm_width = max(0, min(1, tile_norm_width))
                tile_norm_height = max(0, min(1, tile_norm_height))

                if tile_norm_width > 0 and tile_norm_height > 0:
                    tile_annotations.append({
                        'category': ann['category'],
                        'center_x': tile_center_x,
                        'center_y': tile_center_y,
                        'width': tile_norm_width,
                        'height': tile_norm_height
                    })

    return tile_annotations


def create_tiles(image_path, annotation_path, output_dir, tile_width=800, tile_height=1068,
                 overlap=0.1, min_area_ratio=0.3, skip_empty_tiles=True):
    """Create tiles from a large image with corresponding annotations"""

    # Load image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Load annotations
    annotations = load_yolo_annotations(annotation_path, img_width, img_height)

    # Calculate step size (with overlap)
    step_x = int(tile_width * (1 - overlap))
    step_y = int(tile_height * (1 - overlap))

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    tiles_created = 0
    tiles_with_annotations = 0

    # Generate tiles
    for y in range(0, img_height - tile_height + 1, step_y):
        for x in range(0, img_width - tile_width + 1, step_x):
            tile_annotations = get_tile_annotations(
                annotations, x, y, tile_width, tile_height, min_area_ratio)
            if len(tile_annotations) == 0 and skip_empty_tiles:
                continue
            tile = image.crop((x, y, x + tile_width, img_height))

            # Create tile filename
            tile_name = f"{base_name}_tile_{y // step_y:03d}_{x // step_x:03d}"
            tile_image_path = os.path.join(output_dir, f"{tile_name}.jpg")
            tile_annotation_path = os.path.join(output_dir, f"{tile_name}.txt")

            # Save tile image
            tile.save(tile_image_path, "JPEG", quality=95)

            # Save tile annotations
            with open(tile_annotation_path, 'w') as f:
                for ann in tile_annotations:
                    f.write(f"{ann['category']} {ann['center_x']:.6f} {ann['center_y']:.6f} "
                            f"{ann['width']:.6f} {ann['height']:.6f}\n")

            tiles_created += 1
            if len(tile_annotations) > 0:
                tiles_with_annotations += 1

    # Handle edge tiles (right and bottom edges)
    # Right edge tiles
    for y in range(0, img_height - tile_height + 1, step_y):
        x = img_width - tile_width
        if x > 0 and x % step_x != 0:  # Only if not already covered
            tile_annotations = get_tile_annotations(
                annotations, x, y, tile_width, tile_height, min_area_ratio)
            if len(tile_annotations) == 0 and skip_empty_tiles:
                continue
            tile = image.crop((x, y, x + tile_width, img_height))

            tile_name = f"{base_name}_tile_{y // step_y:03d}_edge_right"
            tile_image_path = os.path.join(output_dir, f"{tile_name}.jpg")
            tile_annotation_path = os.path.join(output_dir, f"{tile_name}.txt")

            tile.save(tile_image_path, "JPEG", quality=95)
            with open(tile_annotation_path, 'w') as f:
                for ann in tile_annotations:
                    f.write(f"{ann['category']} {ann['center_x']:.6f} {ann['center_y']:.6f} "
                            f"{ann['width']:.6f} {ann['height']:.6f}\n")

            tiles_created += 1
            if len(tile_annotations) > 0:
                tiles_with_annotations += 1

    # Bottom edge tiles
    for x in range(0, img_width - tile_width + 1, step_x):
        y = img_height - tile_height
        if y > 0 and y % step_y != 0:  # Only if not already covered

            tile_annotations = get_tile_annotations(
                annotations, x, y, tile_width, tile_height, min_area_ratio)
            if len(tile_annotations) == 0 and skip_empty_tiles:
                continue
            tile = image.crop((x, y, x + tile_width, img_height))

            tile_name = f"{base_name}_tile_edge_bottom_{x // step_x:03d}"
            tile_image_path = os.path.join(output_dir, f"{tile_name}.jpg")
            tile_annotation_path = os.path.join(output_dir, f"{tile_name}.txt")

            tile.save(tile_image_path, "JPEG", quality=95)
            with open(tile_annotation_path, 'w') as f:
                for ann in tile_annotations:
                    f.write(f"{ann['category']} {ann['center_x']:.6f} {ann['center_y']:.6f} "
                            f"{ann['width']:.6f} {ann['height']:.6f}\n")

            tiles_created += 1
            if len(tile_annotations) > 0:
                tiles_with_annotations += 1

    # Bottom-right corner tile
    x = img_width - tile_width
    y = img_height - tile_height
    if x > 0 and y > 0:

        tile_annotations = get_tile_annotations(
            annotations, x, y, tile_width, tile_height, min_area_ratio)

        if len(tile_annotations) > 0 or skip_empty_tiles:
            tile = image.crop((x, y, img_width, img_height))



            tile_name = f"{base_name}_tile_edge_corner"
            tile_image_path = os.path.join(output_dir, f"{tile_name}.jpg")
            tile_annotation_path = os.path.join(output_dir, f"{tile_name}.txt")

            tile.save(tile_image_path, "JPEG", quality=95)
            with open(tile_annotation_path, 'w') as f:
                for ann in tile_annotations:
                    f.write(f"{ann['category']} {ann['center_x']:.6f} {ann['center_y']:.6f} "
                            f"{ann['width']:.6f} {ann['height']:.6f}\n")

            tiles_created += 1
            if len(tile_annotations) > 0:
                tiles_with_annotations += 1

    return tiles_created, tiles_with_annotations


def process_dataset(input_dir, output_dir, tile_width=800, tile_height=1068,
                    overlap=0.1, min_area_ratio=0.3):
    """Process entire dataset"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all image files
    image_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            annotation_file = os.path.join(input_dir, file.rsplit('.', 1)[0] + '.txt')
            if os.path.exists(annotation_file):
                image_files.append(file)

    print(f"Found {len(image_files)} images with annotations")

    total_tiles = 0
    total_tiles_with_annotations = 0
    processing_stats = []

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)
        annotation_path = os.path.join(input_dir, image_file.rsplit('.', 1)[0] + '.txt')

        try:
            tiles_created, tiles_with_annotations = create_tiles(
                image_path, annotation_path, output_dir,
                tile_width, tile_height, overlap, min_area_ratio)

            total_tiles += tiles_created
            total_tiles_with_annotations += tiles_with_annotations

            processing_stats.append({
                'image': image_file,
                'tiles_created': tiles_created,
                'tiles_with_annotations': tiles_with_annotations
            })

            print(f"  {image_file}: {tiles_created} tiles ({tiles_with_annotations} with annotations)")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Save processing statistics
    stats = {
        'input_directory': input_dir,
        'output_directory': output_dir,
        'tile_dimensions': {'width': tile_width, 'height': tile_height},
        'overlap': overlap,
        'min_area_ratio': min_area_ratio,
        'total_images_processed': len(image_files),
        'total_tiles_created': total_tiles,
        'total_tiles_with_annotations': total_tiles_with_annotations,
        'processing_details': processing_stats
    }

    with open(os.path.join(output_dir, 'tiling_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nProcessing complete!")
    print(f"Total tiles created: {total_tiles}")
    print(f"Tiles with annotations: {total_tiles_with_annotations}")
    print(f"Coverage: {total_tiles_with_annotations / total_tiles * 100:.1f}% of tiles contain objects")


def main():
    parser = argparse.ArgumentParser(description='Create tiles from large aerial images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing large images and annotations')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save tiles and annotations')
    parser.add_argument('--tile_width', type=int, default=800,
                        help='Width of each tile')
    parser.add_argument('--tile_height', type=int, default=1068,
                        help='Height of each tile')
    parser.add_argument('--overlap', type=float, default=0.1,
                        help='Overlap between tiles (0.0 to 0.5)')
    parser.add_argument('--min_area_ratio', type=float, default=0.3,
                        help='Minimum area ratio to include annotation in tile')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    if args.overlap < 0 or args.overlap > 0.5:
        print("Error: Overlap should be between 0.0 and 0.5")
        return

    if args.min_area_ratio < 0 or args.min_area_ratio > 1:
        print("Error: min_area_ratio should be between 0.0 and 1.0")
        return

    print(f"Tiling configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Tile size: {args.tile_width}x{args.tile_height}")
    print(f"  Overlap: {args.overlap * 100:.1f}%")
    print(f"  Min area ratio: {args.min_area_ratio}")

    process_dataset(args.input_dir, args.output_dir,
                    args.tile_width, args.tile_height,
                    args.overlap, args.min_area_ratio)


if __name__ == '__main__':
    main()