@echo off
title Smart License Plate Detection — Run All
color 0A

echo.
echo ==============================================================
echo   SMART LICENSE PLATE DETECTION — FULL PIPELINE
echo ==============================================================
echo.

:: ──────────────────────────────────────────────────────────────
:: STEP 1 — Install Dependencies
:: ──────────────────────────────────────────────────────────────
echo [STEP 1] Installing dependencies...
echo --------------------------------------------------------------
pip install -r requirements.txt
echo.
echo [OK] Dependencies installed.
echo.
pause

:: ──────────────────────────────────────────────────────────────
:: STEP 2 — Run Evaluation (model metrics + report)
:: ──────────────────────────────────────────────────────────────
echo [STEP 2] Running model evaluation...
echo --------------------------------------------------------------
echo   Generates: evaluation_report.txt + evaluation_output\ images
echo.
python evaluate.py --skip-val
echo.
echo [OK] Evaluation complete. Check evaluation_report.txt
echo.
pause

:: ──────────────────────────────────────────────────────────────
:: STEP 3 — Standalone OCR test on a plate crop
:: ──────────────────────────────────────────────────────────────
echo [STEP 3] Running standalone OCR on a plate crop...
echo --------------------------------------------------------------
echo   Input : plate_crop_test.jpg (auto-cropped plate region)
echo.
python ocr_pipeline.py --image crop_255.jpg
echo.
echo [OK] OCR test complete.
echo.
pause

:: ──────────────────────────────────────────────────────────────
:: STEP 4 — Full End-to-End Pipeline on a test image
:: ──────────────────────────────────────────────────────────────
echo [STEP 4] Running full detection + OCR + database pipeline...
echo --------------------------------------------------------------
echo   Input  : archive/images/test/Cars251.png
echo   Output : detection_output\Cars251_detected.jpg
echo   DB     : plates.db
echo.
python detect_and_recognize.py --source archive/images/test/Cars255.png
echo.
echo [OK] Full pipeline complete.
echo.
pause

:: ──────────────────────────────────────────────────────────────
:: STEP 5 — View database records
:: ──────────────────────────────────────────────────────────────
echo [STEP 5] Viewing database records...
echo --------------------------------------------------------------
python database.py
echo.
echo [OK] Done.
echo.

echo ==============================================================
echo   ALL STEPS COMPLETE!
echo ==============================================================
echo.
echo   Output files:
echo     evaluation_report.txt     - Model metrics report
echo     evaluation_output\        - Annotated sample images
echo     detection_output\         - Full pipeline output images
echo     plates.db                 - SQLite database of detections
echo.
pause
