#!/usr/bin/env bash
# ==============================================================================
# IRIS Location Helper Build Script
# Packages IrisLocationHelper into an authentic macOS .app bundle with Info.plist
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

BUNDLE_DIR="$ROOT_DIR/bin/IrisLocationHelper.app"
MACOS_DIR="$BUNDLE_DIR/Contents/MacOS"
RESOURCES_DIR="$BUNDLE_DIR/Contents/Resources"

echo "🔨 Building IRIS macOS Location Helper bundle..."

# Clean and create directory structure
rm -rf "$BUNDLE_DIR"
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Copy Info.plist
cp "$SCRIPT_DIR/Info.plist" "$BUNDLE_DIR/Contents/Info.plist"

# Compile Swift binary
swiftc -O -parse-as-library \
    -framework CoreLocation \
    -framework Foundation \
    "$SCRIPT_DIR/main.swift" \
    -o "$MACOS_DIR/IrisLocationHelper"

# Ad-hoc sign bundle for macOS TCC registration
codesign --force --deep --sign - "$BUNDLE_DIR"

echo "✅ Successfully built: $BUNDLE_DIR"

