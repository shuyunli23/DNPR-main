import os
import csv
import random
from collections import defaultdict
from PIL import Image

path = '../../all_data/Dataset/BTAD/BTech_Dataset_transformed'
output_file = os.path.join(path, 'btad.csv')

dataset = []
stats = defaultdict(lambda: {
    'train_normal': 0,
    'test_normal': 0,
    'test_anomaly': 0,
    'resolutions': set()
})

print('\n🔍 Scanning BTAD dataset...\n')

for filename in os.listdir(path):
    if not filename.endswith('.csv'):
        cls_name = f'product{filename}'
        print(f'  Processing: {cls_name:<15}', end=' ')

        # ========== Train split (normal only) ==========
        split = 'train'
        label = 'normal'
        mask = ''
        train_path = os.path.join(path, filename, split, 'ok')

        if os.path.exists(train_path):
            files = os.listdir(train_path)
            for file in random.sample(files, len(files)):
                if file.endswith('.bmp') or file.endswith('.png'):
                    image_rel = os.path.join(filename, split, 'ok', file)
                    image_full = os.path.join(path, image_rel)

                    # Read resolution
                    try:
                        with Image.open(image_full) as img:
                            stats[cls_name]['resolutions'].add((img.width, img.height))
                    except Exception as e:
                        print(f"\n  ⚠️  Error reading {image_rel}: {e}")

                    dataset.append([cls_name, split, label, image_rel, mask])
                    stats[cls_name]['train_normal'] += 1

        # ========== Test split ==========
        split = 'test'
        test_base_path = os.path.join(path, filename, split)

        if os.path.exists(test_base_path):
            for test_filename in os.listdir(test_base_path):
                if test_filename == 'ok':
                    # Normal samples
                    label = 'normal'
                    mask = ''
                    test_normal_path = os.path.join(test_base_path, test_filename)
                    files = os.listdir(test_normal_path)
                    for file in random.sample(files, len(files)):
                        if file.endswith('.bmp') or file.endswith('.png'):
                            image_rel = os.path.join(filename, split, test_filename, file)
                            image_full = os.path.join(path, image_rel)

                            try:
                                with Image.open(image_full) as img:
                                    stats[cls_name]['resolutions'].add((img.width, img.height))
                            except:
                                pass

                            dataset.append([cls_name, split, label, image_rel, mask])
                            stats[cls_name]['test_normal'] += 1
                else:
                    # Anomaly samples
                    label = 'anomaly'
                    img_path = os.path.join(test_base_path, test_filename)
                    img_files = os.listdir(img_path)
                    for img_file in random.sample(img_files, len(img_files)):
                        if img_file.endswith('.bmp') or img_file.endswith('.png'):
                            # Handle mask file naming (product 01 needs .bmp -> .png)
                            if filename == '01':
                                msk_file = img_file.replace('.bmp', '.png')
                            else:
                                msk_file = img_file

                            mask_rel = os.path.join(filename, 'ground_truth', test_filename, msk_file)
                            image_rel = os.path.join(filename, split, test_filename, img_file)
                            image_full = os.path.join(path, image_rel)

                            try:
                                with Image.open(image_full) as img:
                                    stats[cls_name]['resolutions'].add((img.width, img.height))
                            except:
                                pass

                            dataset.append([cls_name, split, label, image_rel, mask_rel])
                            stats[cls_name]['test_anomaly'] += 1

        # Print class info
        unique_res = stats[cls_name]['resolutions']
        if len(unique_res) == 1:
            w, h = list(unique_res)[0]
            print(f'✓ Resolution: {w}×{h}')
        elif len(unique_res) > 0:
            print(f'✓ {len(unique_res)} resolutions')
        else:
            print('✓')

# ========== Write CSV ==========
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['object', 'split', 'label', 'image', 'mask'])
    writer.writerows(dataset)

print(f'\n💾 CSV saved to: {output_file}')

# ========== Calculate totals ==========
total_train = 0
total_test_n = 0
total_test_a = 0
all_resolutions = set()

for cls_stats in stats.values():
    total_train += cls_stats['train_normal']
    total_test_n += cls_stats['test_normal']
    total_test_a += cls_stats['test_anomaly']
    all_resolutions.update(cls_stats['resolutions'])

# ========== Print Statistics ==========
print('\n' + '=' * 105)
print('📊 BTAD Dataset Statistics')
print('=' * 105)
print(f'{"Product":<20} {"Train (N)":<12} {"Test (N)":<12} {"Test (A)":<12} {"Total":<12} {"Resolution":<25}')
print('-' * 105)

for cls_name in sorted(stats.keys()):
    train_n = stats[cls_name]['train_normal']
    test_n = stats[cls_name]['test_normal']
    test_a = stats[cls_name]['test_anomaly']
    total = train_n + test_n + test_a

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

    print(f'{cls_name:<20} {train_n:<12} {test_n:<12} {test_a:<12} {total:<12} {res_str:<25}')

print('-' * 105)
print(
    f'{"TOTAL":<20} {total_train:<12} {total_test_n:<12} {total_test_a:<12} {total_train + total_test_n + total_test_a:<12}')
print('=' * 105)

# ========== Print Summary ==========
print('\n📋 Summary:')
print(f'  • Total products: {len(stats)}')
print(f'  • Total images: {total_train + total_test_n + total_test_a}')
print(f'    ├─ Training (normal): {total_train}')
print(f'    └─ Testing: {total_test_n + total_test_a}')
print(f'       ├─ Normal: {total_test_n}')
print(f'       └─ Anomaly: {total_test_a}')

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

print('=' * 105 + '\n')