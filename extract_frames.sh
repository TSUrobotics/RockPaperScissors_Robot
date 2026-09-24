#!/usr/bin/env bash
#
# extract_frames.sh
#
# Usage:
#   ./extract_frames.sh <video_file> [image_format] [ffmpeg_options]
#
#   <video_file>     : Path to the source video (required)
#   [image_format]   : Extension/format for the output images.
#                      Common choices: png, jpg, tif, bmp …
#                      Default = png
#   [ffmpeg_options] : Any extra ffmpeg options you want to pass,
#                      e.g. "-vf fps=2" to limit to 2 frames/sec.
#
# Example:
#   ./extract_frames.sh mymovie.mkv jpg "-vf fps=10"
#
# The script will:
#   1. Verify the input file exists and is readable.
#   2. Create a sub‑directory named "<basename>_frames".
#   3. Run ffmpeg to dump each frame into that directory using a zero‑padded
#      sequential filename (frame_000001.png, frame_000002.png, …).
#
# Requirements:
#   - Bash  (tested with 4.x+)
#   - ffmpeg (any recent build with libavcodec/libavformat)
#
# -------------------------------------------------------------------------

set -euo pipefail   # safer script execution

# -------------------------- Helper functions ---------------------------

print_usage() {
    cat <<EOF
Usage: $(basename "$0") <video_file> [image_format] [ffmpeg_options]

  <video_file>    Path to the source video (required)
  [image_format]  Output image format/extension (default: png)
  [ffmpeg_options]  Extra options passed straight to ffmpeg (optional)

Examples:
  $(basename "$0") movie.mp4
  $(basename "$0") movie.mp4 jpg
  $(basename "$0") movie.mp4 jpg "-vf fps=5"
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

# --------------------------- Argument handling -------------------------

if [[ $# -lt 1 || $# -gt 3 ]]; then
    print_usage
    exit 1
fi

VIDEO_PATH=$1
IMG_FMT=${2:-png}          # default to png if not supplied
FFMPEG_EXTRA=${3:-}        # may be empty

# --------------------------- Sanity checks -----------------------------

[[ -f "$VIDEO_PATH" ]] || die "File not found: $VIDEO_PATH"
[[ -r "$VIDEO_PATH" ]] || die "Cannot read file: $VIDEO_PATH"

# Ensure ffmpeg is available
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found in PATH"

# Strip trailing slash and resolve absolute path (optional but nice)
VIDEO_ABS=$(realpath "$VIDEO_PATH")
VIDEO_DIR=$(dirname "$VIDEO_ABS")
VIDEO_BASE=$(basename "$VIDEO_ABS")
VIDEO_STEM="${VIDEO_BASE%.*}"   # name without extension

# --------------------------- Output directory --------------------------

OUTDIR="${VIDEO_DIR}/${VIDEO_STEM}_frames"

# If the directory already exists, ask what to do
if [[ -d "$OUTDIR" ]]; then
    read -rp "Directory '$OUTDIR' already exists. Overwrite? [y/N] " yn
    case "$yn" in
        [Yy]* )
            rm -rf "$OUTDIR"
            ;;
        * )
            die "Aborted by user."
            ;;
    esac
fi

mkdir -p "$OUTDIR"

# --------------------------- ffmpeg command ---------------------------

# ffmpeg naming pattern:
#   - Use a fixed width (6 digits) zero‑padded sequence.
#   - Example: frame_000001.png
OUTPUT_PATTERN="${OUTDIR}/frame_%06d.${IMG_FMT}"

# Build the full ffmpeg command line.
#   -i "$VIDEO_ABS"               : input video
#   $FFMPEG_EXTRA                : optional user‑supplied options
#   -vsync 0                     : write exactly one image per frame
#   "$OUTPUT_PATTERN"            : output file pattern
FFMPEG_CMD=(
    ffmpeg -hide_banner -loglevel error -stats
    -i "$VIDEO_ABS"
)

# Append any extra arguments supplied by the user (they may contain spaces,
# so we keep them as a single string and let Bash split them later.)
if [[ -n "$FFMPEG_EXTRA" ]]; then
    # eval is safe here because we deliberately want word‑splitting on the
    # extra options the user typed (e.g. "-vf fps=5").
    eval "FFMPEG_CMD+=( $FFMPEG_EXTRA )"
fi

FFMPEG_CMD+=(
    #-vsync 0                # one image per input frame
    "$OUTPUT_PATTERN"
)

# --------------------------- Run ffmpeg ------------------------------

echo "Extracting frames from: $VIDEO_ABS"
echo "Output directory:       $OUTDIR"
echo "Image format:           $IMG_FMT"
[[ -n "$FFMPEG_EXTRA" ]] && echo "Extra ffmpeg opts:      $FFMPEG_EXTRA"
echo "Running: ${FFMPEG_CMD[*]}"

# Execute the assembled command
"${FFMPEG_CMD[@]}"

echo "Done! Frames saved in: $OUTDIR"

