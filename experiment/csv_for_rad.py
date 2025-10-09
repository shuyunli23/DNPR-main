import os
import csv
import random

path = '/remote-home/cs_igps_lsy/pythonProject/all_data/Dataset/RAD'

output_file = os.path.join(path, 'rad.csv')

dataset = []
train_count = 0
n_count = 0
a_count = 0
for filename in os.listdir(path):
    if not filename.endswith('.csv'):
        print(f'Save: {filename}')
        cls_name = filename
        split = 'train'
        label = 'normal'
        mask = ''
        files = os.listdir(os.path.join(path, filename, split, 'good'))
        for file in random.sample(files, len(files)):
            if '(' in file or ')' in file:
                continue
            if file.endswith('.png'):
                image = os.path.join(filename, split, 'good', file)
                dataset.append([cls_name, split, label, image, mask])
                train_count += 1
        split = 'test'
        for test_filename in os.listdir(os.path.join(path, filename, split)):
            if test_filename == 'good':
                label = 'normal'
                mask = ''
                files = os.listdir(os.path.join(path, filename, split, test_filename))
                for file in random.sample(files, len(files)):
                    if '(' in file or ')' in file:
                        continue
                    if file.endswith('.png'):
                        image = os.path.join(filename, split, test_filename, file)
                        dataset.append([cls_name, split, label, image, mask])
                        n_count += 1
            else:
                label = 'anomaly'
                img_path = os.path.join(path, filename, split, test_filename)
                img_files = os.listdir(img_path)
                for img_file in random.sample(img_files, len(img_files)):
                    if '(' in img_file or ')' in img_file:
                        continue
                    if img_file.endswith('.png'):
                        msk_file = img_file
                        mask = os.path.join(filename, 'ground_truth', test_filename, msk_file)
                        if os.path.isfile(os.path.join(path, mask)):
                            image = os.path.join(filename, split, test_filename, img_file)
                            dataset.append([cls_name, split, label, image, mask])
                            a_count += 1
print(f'Overall train samples: {int(train_count / 4)}'
      f'\tOverall test samples: {int(n_count / 4 + a_count)}[N/A-{int(n_count / 4)}/{a_count}]')
print(f'Overall samples: {int(train_count / 4 + n_count / 4 + a_count)}')

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['object', 'split', 'label', 'image', 'mask'])  # write in column headings
    writer.writerows(dataset)  # write to dataset


