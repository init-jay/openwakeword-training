#!/usr/bin/env bash
#
# Download the microWakeWord ambient negative sets. ~5.7 GB.
#
# THIS DELIBERATELY DOWNLOADS NOTHING ELSE. mWW's augmentation wants impulse
# responses and background audio, and those are the SAME three corpora setup-data.sh
# already fetched for openWakeWord - MIT RIRs, AudioSet and FMA, from the same URLs.
# mww/config.py points `impulse_paths` and `background_paths` at them in place.
# Re-fetching would cost another ~10 GB for identical bytes.
#
# What is genuinely new is the ambient negative sets. They are pre-computed
# RaggedMmap spectrograms in microWakeWord's own feature format, so unlike the audio
# corpora they cannot be shared with the openWakeWord side, and unlike this repo's
# adversarial negatives they are large and general - they are what teaches the model
# that ordinary rooms, music and conversation are not the wake word.
#
# They also carry mWW's primary metric. Its model selection minimises false accepts
# per hour on ambient audio before maximising recall, and testing_ambient /
# validation_ambient are what that is measured on.
#
# Idempotent: each set is skipped if already unpacked.
#
#   ./setup-mww-data.sh
#   DATA_DIR=/mnt/big/data ./setup-mww-data.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
AMBIENT_DIR="$DATA_DIR/mww_ambient"
BASE_URL="https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main"

# name:approx-size, for the progress message only.
SETS=(
    "dinner_party:444MB"
    "dinner_party_eval:82MB"
    "no_speech:2.0GB"
    "speech:3.2GB"
)

mkdir -p "$AMBIENT_DIR"

# Warn early rather than at training time: mWW needs these and this script does not
# provide them on purpose.
missing=""
for d in mit_rirs audioset_16k fma; do
    [ -d "$DATA_DIR/$d" ] || missing="$missing $d"
done
if [ -n "$missing" ]; then
    echo "WARNING: missing from $DATA_DIR:$missing"
    echo "         These are the augmentation corpora, shared with the openWakeWord"
    echo "         side. Run ./setup-data.sh - this script does not duplicate them."
    echo
fi

for entry in "${SETS[@]}"; do
    name="${entry%%:*}"
    size="${entry##*:}"
    target="$AMBIENT_DIR/$name"

    if [ -d "$target" ]; then
        echo "=== $name already present, skipping"
        continue
    fi

    echo "=== downloading $name ($size)"
    curl -L --fail -o "$AMBIENT_DIR/$name.zip" "$BASE_URL/$name.zip"

    echo "=== unpacking $name"
    mkdir -p "$target"
    unzip -q "$AMBIENT_DIR/$name.zip" -d "$target"
    rm -f "$AMBIENT_DIR/$name.zip"
done

echo
echo "=== done. RaggedMmap sets found:"
# The config wants a features_dir whose children are the split directories
# (training/, validation/, testing/, testing_ambient/, validation_ambient/), each
# containing *_mmap/ directories. Report what actually landed rather than assuming
# the archive layout - it is what the --ambient arguments have to point at.
found=0
while IFS= read -r mmap; do
    parent="$(dirname "$(dirname "$mmap")")"
    echo "  $parent"
    found=1
done < <(find "$AMBIENT_DIR" -type d -name "*_mmap" 2>/dev/null | head -40 | sort -u)

if [ "$found" -eq 0 ]; then
    echo "  NONE FOUND - the archives did not contain *_mmap/ directories."
    echo "  Inspect $AMBIENT_DIR before configuring a run; data.py globs"
    echo "  <features_dir>/{training,validation,testing,...}/**/*_mmap/"
    exit 1
fi

echo
echo "Pass the deduplicated parents to mww/config.py --ambient, e.g."
echo "  python -m mww.config --wake-word \"hey seeree\" \\"
echo "      --ambient $AMBIENT_DIR/speech $AMBIENT_DIR/no_speech \\"
echo "      --data-dir $DATA_DIR --out training_parameters.yaml"
