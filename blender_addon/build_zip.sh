#!/bin/bash
# Build the installable Blender addon zip.
#
# Usage:  bash blender_addon/build_zip.sh
# Output: blender_addon/topmod_blender.zip
#
# Install in Blender:
#   Edit → Preferences → Add-ons → Install from Disk → topmod_blender.zip

set -euo pipefail
cd "$(dirname "$0")"

# Sync topmod core from the repo (excluding torch-dependent modules)
DEST=topmod_blender/topmod
mkdir -p "$DEST"
for f in __init__.py dlfl.py operators.py primitives.py validate.py \
         io.py high_level_ops.py subdivision.py remeshing.py; do
    cp "../topmod/$f" "$DEST/$f"
done

# Patch: strip tokenizer import from bundled __init__.py
python3 -c "
t = open('$DEST/__init__.py').read()
# Remove tokenizer import block
import re
t = re.sub(r'from \.tokenizer import.*?\)', '', t, flags=re.DOTALL)
# Remove tokenizer from __all__
t = re.sub(r'    # Tokenizer\n.*?\]', ']\n', t, flags=re.DOTALL)
open('$DEST/__init__.py', 'w').write(t)
" 2>/dev/null || true

# Remove __pycache__
find topmod_blender -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Build zip
rm -f topmod_blender.zip
cd ..
zip -r blender_addon/topmod_blender.zip \
    blender_addon/topmod_blender/ \
    -x '*.pyc' -x '*__pycache__*'

echo ""
echo "Built: blender_addon/topmod_blender.zip"
echo "Install: Blender → Preferences → Add-ons → Install from Disk"
