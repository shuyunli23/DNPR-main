import os
import csv
import random
from collections import defaultdict
from PIL import Image

path = '/mnt/igps_622/lsy/Project/all_data/Dataset/CableInspect_AD'
output_file = os.path.join(path, 'cableinspect_ad.csv')

dataset = []
stats = defaultdict(lambda: {
    'normal': 0,
    'anomaly': 0,
    'resolutions': set()
})

print('\n🔍 Scanning CableInspect-AD dataset...\n')

for filename in os.listdir(path):
    if os.path.isdir(os.path.join(path, filename)) and filename != '__pycache__':
        cls_name = filename.lower()
        print(f'  Processing: {cls_name:<20}', end=' ')

        images_path = os.path.join(path, filename, 'images')
        if not os.path.exists(images_path):
            print('❌ No images directory')
            continue

        for test_filename in os.listdir(images_path):
            img_dir = os.path.join(images_path, test_filename)

            if not os.path.isdir(img_dir):
                continue

            img_files = os.listdir(img_dir)
            for img_file in random.sample(img_files, len(img_files)):
                if img_file.endswith('.png'):
                    img_rel_path = os.path.join(filename, 'images', test_filename, img_file)
                    img_full_path = os.path.join(path, img_rel_path)

                    # Read resolution
                    try:
                        with Image.open(img_full_path) as img:
                            stats[cls_name]['resolutions'].add((img.width, img.height))
                    except Exception as e:
                        print(f"\n  ⚠️  Error reading {img_rel_path}: {e}")

                    # Check if mask exists
                    msk_file = img_file
                    msk_path = os.path.join(filename, 'masks', test_filename)
                    msk_full_path = os.path.join(path, msk_path, msk_file)

                    if os.path.exists(msk_full_path):
                        # Anomaly sample
                        label = 'anomaly'
                        dataset.append([cls_name, 'test', label, img_rel_path,
                                        os.path.join(msk_path, msk_file)])
                        stats[cls_name]['anomaly'] += 1
                    else:
                        # Normal sample (added to both train and test)
                        label = 'normal'
                        dataset.append([cls_name, 'train', label, img_rel_path, ''])
                        dataset.append([cls_name, 'test', label, img_rel_path, ''])
                        stats[cls_name]['normal'] += 1

        # Print class info
        n_count = stats[cls_name]['normal']
        a_count = stats[cls_name]['anomaly']
        unique_res = stats[cls_name]['resolutions']

        if len(unique_res) == 1:
            w, h = list(unique_res)[0]
            print(f'✓ N: {n_count:<4} A: {a_count:<4} | Resolution: {w}×{h}')
        elif len(unique_res) > 0:
            print(f'✓ N: {n_count:<4} A: {a_count:<4} | {len(unique_res)} resolutions')
        else:
            print(f'✓ N: {n_count:<4} A: {a_count:<4}')

# ========== Write CSV ==========
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['object', 'split', 'label', 'image', 'mask'])
    writer.writerows(dataset)

print(f'\n💾 CSV saved to: {output_file}')

# ========== Calculate totals ==========
total_normal = 0
total_anomaly = 0
all_resolutions = set()

for cls_stats in stats.values():
    total_normal += cls_stats['normal']
    total_anomaly += cls_stats['anomaly']
    all_resolutions.update(cls_stats['resolutions'])

# ========== Print Statistics ==========
print('\n' + '=' * 100)
print('📊 CableInspect-AD Dataset Statistics')
print('=' * 100)
print(f'{"Category":<20} {"Normal":<12} {"Anomaly":<12} {"Total":<12} {"Resolution":<25}')
print('-' * 100)

for cls_name in sorted(stats.keys()):
    n_count = stats[cls_name]['normal']
    a_count = stats[cls_name]['anomaly']
    total = n_count + a_count

    # Format resolution
    resolutions = stats[cls_name]['resolutions']
    if len(resolutions) == 1:
        w, h = list(resolutions)[0]
        res_str = f"{w}×{h}"
    elif len(resolutions) > 1:
        sorted_res = sorted(resolutions, key=lambda x: x[0] * x[1])
        w1, h1 = sorted_res[0]
        w2, h2 = sorted_res[-1]
        res_str = f"{w1}×{h1} - {w2}×{h2}"
    else:
        res_str = "N/A"

    print(f'{cls_name:<20} {n_count:<12} {a_count:<12} {total:<12} {res_str:<25}')

print('-' * 100)
print(f'{"TOTAL (Unique)":<20} {total_normal:<12} {total_anomaly:<12} {total_normal + total_anomaly:<12}')
print('=' * 100)

# ========== Print Summary ==========
print('\n📋 Summary (Unique Sample Counts):')
print(f'  • Total categories: {len(stats)}')
print(f'  • Total unique samples: {total_normal + total_anomaly}')
print(f'    ├─ Normal: {total_normal}')
print(f'    └─ Anomaly: {total_anomaly}')

print(f'\n  💡 Dataset Organization Note:')
print(f'    • Normal samples are used in BOTH train and test splits')
print(f'    • Total CSV rows: {len(dataset)}')
print(f'      ├─ Training entries: {total_normal} (all normal)')
print(f'      └─ Testing entries: {total_normal + total_anomaly}')
print(f'         ├─ Normal: {total_normal}')
print(f'         └─ Anomaly: {total_anomaly}')

# Resolution info
if len(all_resolutions) > 0:
    print(f'\n  • Resolution Statistics:')
    if len(all_resolutions) == 1:
        w, h = list(all_resolutions)[0]
        print(f'    └─ All images: {w}×{h}')
    else:
        sorted_all = sorted(all_resolutions, key=lambda x: x[0] * x[1])
        print(f'    ├─ Unique resolutions: {len(all_resolutions)}')
        for w, h in sorted_all:
            print(f'    │  • {w}×{h}')

print('=' * 100 + '\n')