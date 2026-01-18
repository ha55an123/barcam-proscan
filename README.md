# Barcam ProScan v2.0

Industrial-grade multi-format barcode scanner with AI defect detection and ISO 15415 compliance.

## Supported Barcode Formats

### 1D Barcodes (Linear)
- **EAN-8** - European Article Number (8 digits)
- **EAN-13** - European Article Number (13 digits)
- **UPC-A** - Universal Product Code (12 digits)
- **UPC-E** - Compact UPC (6 digits)
- **Code 39** - Alphanumeric barcode
- **Code 93** - Compact alphanumeric
- **Code 128** - High-density alphanumeric
- **ITF** - Interleaved 2 of 5
- **Codabar** - Libraries, blood banks, logistics

### 2D Barcodes (Matrix)
- **QR Code** - Quick Response codes (most common)
- **Data Matrix** - ECC 200, industrial applications
- **PDF417** - Portable Data File (2D stacked)
- **Aztec Code** - High-density 2D matrix

## Key Features
✅ **Multi-Format Detection** - Automatically detects all barcode types
✅ **AI Defect Detection** - Identifies blur, low contrast, and broken edges
✅ **ISO 15415 Grading** - Grades A, B, C, D, F quality ratings
✅ **Real-time Processing** - 5-60 FPS adjustable scanning
✅ **Statistics Dashboard** - Live tracking of scans, defects, pass rates
✅ **Duplicate Detection** - Configurable time window to prevent re-scans
✅ **Export Options** - CSV for table data, Excel for ISO reports
✅ **Auto-Export** - Optional automatic report generation
✅ **Dark/Light Themes** - User-friendly interface customization
✅ **Snapshot Storage** - Organized by order number

## System Requirements
- **OS:** Windows 10/11, Linux, macOS
- **Camera:** USB Webcam (640x480 minimum, 1280x720 recommended)
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 500MB minimum (2GB recommended)
- **Python:** 3.8-3.12 (if running from source)

## Quick Start
1. Connect USB camera
2. Launch Barcam ProScan
3. Select camera and resolution
4. Click "Start Camera"
5. Present barcode to camera
6. View results in real-time table

## Industrial Use Cases
- **Manufacturing** - Quality control inspection
- **Warehousing** - Inventory tracking and verification
- **Logistics** - Package scanning and sorting
- **Retail** - Product verification and checkout
- **Healthcare** - Medical device tracking (Data Matrix)
- **Automotive** - Parts traceability (Code 128, Data Matrix)
- **Aerospace** - Component identification (Data Matrix)
- **Pharmaceuticals** - Drug verification (Data Matrix, QR)

## Support & Documentation
- **Email:** hairfan545@gmail.com
- **Documentation:** 
- **Issue Tracker:** https://github.com/barcam/proscan/issues

## License
MIT License - See LICENSE.txt for details

---

**Built with:** Python, OpenCV, PyQt5, pyzbar, pandas
**Developer:** Hassan | © 2026 Barcam Technologies


pyinstaller --onefile \
    --windowed \
    --name "Barcam-ProScan" \
    --icon=app.ico \
    --add-data "README.md;." \
    --hidden-import pyzbar \
    --hidden-import cv2 \
    --hidden-import pandas \
    Barcam-proscan.py
