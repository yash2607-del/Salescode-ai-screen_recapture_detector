"""
Script to extract frames from a recorded screen-recapture video at 1-second intervals.
Saves extracted frames into dataset/screen/ using OpenCV.
"""
import sys
from pathlib import Path

# Try importing OpenCV and provide a helpful error if it's missing
try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Please install it using: pip install opencv-python", file=sys.stderr)
    sys.exit(1)

def extract_frames(video_path: Path, output_dir: Path):
    """
    Extracts frames from the video at 1 frame per second.
    """
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
        return

    # Get FPS (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    print(f"Loaded video: {video_path.name}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration_sec:.2f} seconds")
    print(f"Extracting 1 frame every 1.0 second (every {int(round(fps))} frames)...")

    frame_interval = int(round(fps))
    if frame_interval <= 0:
        frame_interval = 30  # Default fallback if FPS detection fails

    extracted_count = 0
    frame_idx = 0
    video_name = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract frame if index is at the correct second mark
        if frame_idx % frame_interval == 0:
            second_mark = frame_idx // frame_interval
            dest_filename = f"extracted_{video_name}_sec_{second_mark:03d}.jpg"
            dest_path = output_dir / dest_filename
            
            # Prevent overwriting
            counter = 1
            while dest_path.exists():
                dest_filename = f"extracted_{video_name}_sec_{second_mark:03d}_{counter}.jpg"
                dest_path = output_dir / dest_filename
                counter += 1
                
            # Save frame
            cv2.imwrite(str(dest_path), frame)
            extracted_count += 1
            
        frame_idx += 1

    cap.release()
    print("\nExtraction complete!")
    print(f"Total extracted and saved frames: {extracted_count}")

def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "dataset" / "screen"

    # Check for command line argument
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        # Scan local directory for video files
        video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
        video_files = []
        for file in base_dir.glob("*"):
            if file.suffix.lower() in video_extensions:
                video_files.append(file)
                
        if not video_files:
            print("No video files found in the root directory.")
            print("Usage: python frame_extractor.py path/to/your/recorded_video.mp4")
            sys.exit(1)
            
        # Select the first video found
        video_path = video_files[0]
        print(f"Auto-detected video file: {video_path.name}")
        if len(video_files) > 1:
            print("Other videos found. You can explicitly run:")
            for vf in video_files[1:]:
                print(f"  python frame_extractor.py {vf.name}")

    extract_frames(video_path, output_dir)

if __name__ == "__main__":
    main()
