import os
import csv
import random

path = '/remote-home/cs_igps_lsy/pythonProject/all_data/Dataset/DTD_Synthetic'

output_file = os.path.join(path, 'dtd_synthetic.csv')

dataset = []

for filename in os.listdir(path):
    if not filename.endswith('.csv'):
        print(f'Save: {filename}')
        cls_name = filename.lower()
        split = 'train'
        label = 'normal'
        mask = ''
        files = os.listdir(os.path.join(path, filename, split, 'good'))
        for file in random.sample(files, len(files)):
            if file.endswith('.png'):
                image = os.path.join(filename, split, 'good', file)
                dataset.append([cls_name, split, label, image, mask])
        split = 'test'
        for test_filename in os.listdir(os.path.join(path, filename, split)):
            if test_filename == 'good':
                label = 'normal'
                mask = ''
                files = os.listdir(os.path.join(path, filename, split, test_filename))
                for file in random.sample(files, len(files)):
                    if file.endswith('.png'):
                        image = os.path.join(filename, split, test_filename, file)
                        dataset.append([cls_name, split, label, image, mask])
            else:
                label = 'anomaly'
                img_path = os.path.join(path, filename, split, test_filename)
                img_files = os.listdir(img_path)
                for img_file in random.sample(img_files, len(img_files)):
                    if img_file.endswith('.png'):
                        name, extension = os.path.splitext(img_file)
                        msk_file = f'{name}_mask' + extension
                        mask = os.path.join(filename, 'ground_truth', test_filename, msk_file)
                        image = os.path.join(filename, split, test_filename, img_file)
                        dataset.append([cls_name, split, label, image, mask])

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['object', 'split', 'label', 'image', 'mask'])  # write in column headings
    writer.writerows(dataset)  # write to dataset


