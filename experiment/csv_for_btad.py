import os
import csv
import random

path = '../../all_data/Dataset/BTAD/BTech_Dataset_transformed'

output_file = os.path.join(path, 'btad.csv')

dataset = []

for filename in os.listdir(path):
    if not filename.endswith('.csv'):
        print(f'Save: {filename}')
        cls_name = 'product{}'.format(filename)
        split = 'train'
        label = 'normal'
        mask = ''
        files = os.listdir(os.path.join(path, filename, split, 'ok'))
        for file in random.sample(files, len(files)):
            if file.endswith('.bmp') or file.endswith('.png'):
                image = os.path.join(filename, split, 'ok', file)
                dataset.append([cls_name, split, label, image, mask])
        split = 'test'
        for test_filename in os.listdir(os.path.join(path, filename, split)):
            if test_filename == 'ok':
                label = 'normal'
                mask = ''
                files = os.listdir(os.path.join(path, filename, split, test_filename))
                for file in random.sample(files, len(files)):
                    if file.endswith('.bmp') or file.endswith('.png'):
                        image = os.path.join(filename, split, test_filename, file)
                        dataset.append([cls_name, split, label, image, mask])
            else:
                label = 'anomaly'
                img_path = os.path.join(path, filename, split, test_filename)
                img_files = os.listdir(img_path)
                for img_file in random.sample(img_files, len(img_files)):
                    if img_file.endswith('.bmp') or img_file.endswith('.png'):
                        if filename == '01':
                            msk_file = img_file.replace('.bmp', '.png')
                        else:
                            msk_file = img_file
                        mask = os.path.join(filename, 'ground_truth', test_filename, msk_file)
                        image = os.path.join(filename, split, test_filename, img_file)
                        dataset.append([cls_name, split, label, image, mask])

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['object', 'split', 'label', 'image', 'mask'])  # write in column headings
    writer.writerows(dataset)  # write to dataset


