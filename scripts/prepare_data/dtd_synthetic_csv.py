import os
import csv
import random
from collections import defaultdict
from PIL import Image

path = '/mnt/igps_622/lsy/Project/all_data/Dataset/DTD_Synthetic/DTD-Synthetic'
output_file = os.path.join(path, 'dtd_synthetic.csv')

dataset = []
stats = defaultdict(lambda: {
    'train_normal': 0,
    'test_normal': 0,
    'test_anomaly': 0,
    'resolutions': set()  # Store unique resolutions
})

print('\n🔍 Scanning dataset and analyzing resolutions...\n')

for filename in os.listdir(path):
    if not filename.endswith('.csv'):
        print(f'Processing: {filename}')
        cls_name = filename.lower()

        # ========== Train split (normal only) ==========
        split = 'train'
        label = 'normal'
        mask = ''
        train_path = os.path.join(path, filename, split, 'good')
        if os.path.exists(train_path):
            files = os.listdir(train_path)
            for file in random.sample(files, len(files)):
                if file.endswith('.png'):
                    image_rel = os.path.join(filename, split, 'good', file)
                    image_full = os.path.join(path, image_rel)

                    # Read image resolution
                    try:
                        with Image.open(image_full) as img:
                            resolution = f"{img.width}×{img.height}"
                            stats[cls_name]['resolutions'].add(resolution)
                    except Exception as e:
                        print(f"  ⚠️  Error reading {image_rel}: {e}")

                    dataset.append([cls_name, split, label, image_rel, mask])
                    stats[cls_name]['train_normal'] += 1

        # ========== Test split ==========
        split = 'test'
        test_base_path = os.path.join(path, filename, split)
        if os.path.exists(test_base_path):
            for test_filename in os.listdir(test_base_path):
                if test_filename == 'good':
                    # Normal samples
                    label = 'normal'
                    mask = ''
                    test_normal_path = os.path.join(test_base_path, test_filename)
                    files = os.listdir(test_normal_path)
                    for file in random.sample(files, len(files)):
                        if file.endswith('.png'):
                            image_rel = os.path.join(filename, split, test_filename, file)
                            image_full = os.path.join(path, image_rel)

                            # Read image resolution
                            try:
                                with Image.open(image_full) as img:
                                    resolution = f"{img.width}×{img.height}"
                                    stats[cls_name]['resolutions'].add(resolution)
                            except Exception as e:
                                print(f"  ⚠️  Error reading {image_rel}: {e}")

                            dataset.append([cls_name, split, label, image_rel, mask])
                            stats[cls_name]['test_normal'] += 1
                else:
                    # Anomaly samples
                    label = 'anomaly'
                    img_path = os.path.join(test_base_path, test_filename)
                    img_files = os.listdir(img_path)
                    for img_file in random.sample(img_files, len(img_files)):
                        if img_file.endswith('.png'):
                            name, extension = os.path.splitext(img_file)
                            msk_file = f'{name}_mask' + extension
                            mask_rel = os.path.join(filename, 'ground_truth', test_filename, msk_file)
                            image_rel = os.path.join(filename, split, test_filename, img_file)
                            image_full = os.path.join(path, image_rel)

                            # Read image resolution
                            try:
                                with Image.open(image_full) as img:
                                    resolution = f"{img.width}×{img.height}"
                                    stats[cls_name]['resolutions'].add(resolution)
                            except Exception as e:
                                print(f"  ⚠️  Error reading {image_rel}: {e}")

                            dataset.append([cls_name, split, label, image_rel, mask_rel])
                            stats[cls_name]['test_anomaly'] += 1

# ========== Write CSV ==========
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['object', 'split', 'label', 'image', 'mask'])
    writer.writerows(dataset)

print(f'\n💾 CSV saved to: {output_file}')

# ========== Print Statistics ==========
print('\n' + '=' * 100)
print('📊 DTD-Synthetic Dataset Statistics')
print('=' * 100)
print(f'{"Class":<20} {"Train (N)":<12} {"Test (N)":<12} {"Test (A)":<12} {"Total":<12} {"Resolution(s)":<25}')
print('-' * 100)

total_train_normal = 0
total_test_normal = 0
total_test_anomaly = 0
all_resolutions = set()

for cls_name in sorted(stats.keys()):
    train_n = stats[cls_name]['train_normal']
    test_n = stats[cls_name]['test_normal']
    test_a = stats[cls_name]['test_anomaly']
    total = train_n + test_n + test_a

    # Format resolutions
    resolutions = stats[cls_name]['resolutions']
    all_resolutions.update(resolutions)

    if len(resolutions) == 1:
        res_str = list(resolutions)[0]
    else:
        # Sort resolutions by total pixels for better display
        sorted_res = sorted(resolutions, key=lambda x: int(x.split('×')[0]) * int(x.split('×')[1]))
        res_str = f"{sorted_res[0]} - {sorted_res[-1]}"

    print(f'{cls_name:<20} {train_n:<12} {test_n:<12} {test_a:<12} {total:<12} {res_str:<25}')

    total_train_normal += train_n
    total_test_normal += test_n
    total_test_anomaly += test_a

print('-' * 100)
print(
    f'{"TOTAL":<20} {total_train_normal:<12} {total_test_normal:<12} {total_test_anomaly:<12} {total_train_normal + total_test_normal + total_test_anomaly:<12}')
print('=' * 100)

# ========== Print Summary ==========
print('\n📋 Summary:')
print(f'  • Total classes: {len(stats)}')
print(f'  • Total training samples: {total_train_normal} (all normal)')
print(f'  • Total test samples: {total_test_normal + total_test_anomaly}')
print(f'    - Normal: {total_test_normal}')
print(f'    - Anomaly: {total_test_anomaly}')
print(f'  • Grand total: {total_train_normal + total_test_normal + total_test_anomaly}')
print(f'\n  • Unique resolutions found: {len(all_resolutions)}')
if len(all_resolutions) <= 10:
    print(f'    {", ".join(sorted(all_resolutions, key=lambda x: int(x.split("×")[0]) * int(x.split("×")[1])))}')
else:
    sorted_res = sorted(all_resolutions, key=lambda x: int(x.split('×')[0]) * int(x.split('×')[1]))
    print(f'    Range: {sorted_res[0]} to {sorted_res[-1]}')
    print(f'    (Total {len(all_resolutions)} different resolutions)')
print('=' * 100 + '\n')