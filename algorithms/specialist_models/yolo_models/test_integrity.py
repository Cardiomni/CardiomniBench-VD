"""
Quick YOLO model loading test (without ultralytics library)
Tests if downloaded models are valid PyTorch archives
"""

import zipfile
from pathlib import Path

def test_yolo_integrity():
    print("=" * 60)
    print("YOLO Model Integrity Test")
    print("=" * 60)

    yolo_dir = Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models/yolo_models")

    models = {
        'yolov8x.pt': 'Stenosis Detection',
        'yolov8x-seg.pt': 'Vessel Segmentation',
        'yolov9c.pt': 'Stenosis Quantification'
    }

    results = {}

    for model_file, purpose in models.items():
        model_path = yolo_dir / model_file

        print(f"\n[Testing] {model_file} ({purpose})")
        print(f"   Path: {model_path}")

        if not model_path.exists():
            print(f"   ❌ File not found")
            results[model_file] = 'missing'
            continue

        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")

        # Check if it's a valid ZIP archive (PyTorch models are ZIP containers)
        try:
            with zipfile.ZipFile(model_path, 'r') as z:
                files = z.namelist()
                print(f"   ✅ Valid ZIP archive ({len(files)} files)")

                # Check for key PyTorch model files
                has_data_pkl = any('data.pkl' in f for f in files)
                has_model = any('model' in f.lower() for f in files)

                if has_data_pkl or has_model:
                    print(f"   ✅ Contains PyTorch model data")
                    results[model_file] = 'valid'
                else:
                    print(f"   ⚠️  ZIP valid but no model data found")
                    results[model_file] = 'incomplete'

        except zipfile.BadZipFile:
            print(f"   ❌ Invalid ZIP archive (corrupted download)")
            results[model_file] = 'corrupted'
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[model_file] = 'error'

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    valid_count = sum(1 for v in results.values() if v == 'valid')
    total_count = len(models)

    for model_file, status in results.items():
        status_symbol = {
            'valid': '✅',
            'incomplete': '⚠️ ',
            'corrupted': '❌',
            'missing': '❌',
            'error': '❌'
        }.get(status, '?')

        print(f"{status_symbol} {model_file}: {status}")

    print(f"\nValid models: {valid_count}/{total_count}")

    if valid_count == total_count:
        print("\n🎉 All YOLO models ready for toolkit integration!")
        return True
    else:
        print("\n⚠️  Some models need re-download")
        return False


if __name__ == "__main__":
    success = test_yolo_integrity()
    exit(0 if success else 1)
