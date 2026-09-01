#!/usr/bin/env python3
"""Convert MP4 video to WebM format for browser compatibility."""

import sys
from pathlib import Path

def convert_video(input_path: str, output_path: str = None):
    """Convert video to WebM format."""
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    if output_path is None:
        output_path = input_path.with_suffix('.webm')
    else:
        output_path = Path(output_path)

    print(f"Converting {input_path.name} to WebM format...")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    try:
        import imageio

        print("Reading video...")
        reader = imageio.get_reader(str(input_path))
        fps = reader.get_meta_data()['fps']

        print(f"FPS: {fps}")
        print("Writing WebM file (this may take a while)...")

        writer = imageio.get_writer(str(output_path), fps=fps, codec='libvpx-vp9', pixelformat='yuv420p')

        frame_count = 0
        for frame in reader:
            writer.append_data(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")

        writer.close()
        reader.close()

        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Conversion complete!")
        print(f"Output file size: {output_size_mb:.1f} MB")
        print(f"Video is now browser-compatible: {output_path}")

        return True

    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nTroubleshooting:")
        print("- Make sure the input file is a valid video")
        print("- Try using a simpler approach with opencv")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_video_to_webm.py <input_video> [output_video]")
        print("\nExample:")
        print("  python convert_video_to_webm.py results/IMG_2013_final/visualization/sidebyside.mp4")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_video(input_file, output_file)
    sys.exit(0 if success else 1)
