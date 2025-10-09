import os
import csv
import random
from PIL import Image

path = '/remote-home/cs_igps_lsy/pythonProject/all_data/Dataset/CableInspect_AD'

output_file = os.path.join(path, 'cableinspect_ad.csv')

dataset = []
t_n_count = 0
t_a_count = 0
for filename in os.listdir(path):
    if os.path.isdir(os.path.join(path, filename)):

        print(f'Save: {filename}')
        cls_name = filename.lower()
        n_count = 0
        a_count = 0
        for test_filename in os.listdir(os.path.join(path, filename, 'images')):

            img_path = os.path.join(filename, 'images', test_filename)
            img_files = os.listdir(os.path.join(path, img_path))
            for img_file in random.sample(img_files, len(img_files)):
                if img_file.endswith('.png'):
                    msk_file = img_file
                    msk_path = os.path.join(filename, 'masks', test_filename)
                    if os.path.exists(os.path.join(path, msk_path, msk_file)):
                        label = 'anomaly'
                        dataset.append([cls_name, 'test', label, os.path.join(img_path, img_file),
                                        os.path.join(msk_path, msk_file)])
                        a_count += 1
                    else:
                        label = 'normal'
                        dataset.append([cls_name, 'train', label, os.path.join(img_path, img_file), ''])
                        dataset.append([cls_name, 'test', label, os.path.join(img_path, img_file), ''])
                        n_count += 1

        print(f'Normal samples: {n_count}\tAnomaly samples: {a_count}')
        t_n_count += n_count
        t_a_count += a_count
print(f'Overall normal samples: {t_n_count}\tOverall anomaly samples: {t_a_count}')
print(f'Overall samples: {t_n_count + t_a_count}')

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['object', 'split', 'label', 'image', 'mask'])  # write in column headings
    writer.writerows(dataset)  # write to dataset


