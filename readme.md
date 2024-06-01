**English** | [**简体中文**](readme-chs.md)

# LabelDet

This tool aims to provide a simple and efficient image annotation solution, suitable for YOLO format object detection tasks.

![demo](./data/zidane.jpg)

It is easy to distribute with only `opencv` package , making it convenient for PyInstaller packaging and distribution.

## Introduction

LabelDet is an image annotation tool designed to simplify and accelerate the image annotation process using the YOLO format. With this tool, users can easily draw annotation boxes on images, edit, and save annotation data.

### Features Overview

- **Annotation Box Editing**: Supports classifying and deleting already drawn annotation boxes.
- **Zoom and Pan**: Supports image zooming and panning for precise annotation.
- **Annotation Saving**: Allows saving annotation data in YOLO format text files.
- **Visual Modes**: Provides various image enhancement modes for better target recognition and annotation.
- **Progress Archiving**: Can archive progress, facilitating the ability to close and resume progress at any time.

The tool offers multiple visual modes to assist with annotation, selectable via the `mode` slider:

- **Mode 1**: Histogram equalization
- **Mode 2/3**: Contrast enhancement
- **Mode 4**: Decrease contrast
- **Mode 5**: Anti-exposure processing
- **Mode 6**: Power law transformation

## Packing command

```bash
pyinstaller -F --version-file version.txt detect_label.py
```
