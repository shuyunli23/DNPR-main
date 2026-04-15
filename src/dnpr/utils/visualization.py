import os, cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dnpr.utils.helpers import denormalization


def plot_segmentation_images(test_img, scores, gts, threshold, save_dir, class_name, plot_more=''):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask <= threshold] = 0
        high_scores = mask * 255
        mask[mask > threshold] = 1
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        alpha = np.where(high_scores == 0, 0.0, 0.3)
        ax_img[4].imshow(vis_img)
        ax_img[4].imshow(high_scores, cmap='hot', alpha=alpha, interpolation='none')
        ax_img[4].title.set_text('Segmentation result')

        plt.axis('off')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()

        if plot_more != '':
            fig_img_another, ax_img_another = plt.subplots(1, 4, figsize=(9, 3))
            fig_img.subplots_adjust(right=0.9)
            for ax_j in ax_img_another:
                ax_j.axes.xaxis.set_visible(False)
                ax_j.axes.yaxis.set_visible(False)
            ax_img_another[0].imshow(img)
            ax_img_another[0].title.set_text('Image')
            ax_img_another[1].imshow(gt, cmap='gray')
            ax_img_another[1].title.set_text('GroundTruth')
            ax_img_another[2].imshow(mask, cmap='gray')
            ax_img_another[2].title.set_text('Predicted mask')

            # Create an array of the same size as the original image to store the superimposed image
            overlay = np.zeros_like(img)
            # Set false positives to red
            overlay[np.logical_and(gt != 1, mask == 255)] = [255, 0, 0]
            # Set true positive as green
            overlay[np.logical_and(gt == 1, mask == 255)] = [0, 255, 0]
            # Set false negatives to blue
            overlay[np.logical_and(gt == 1, mask != 255)] = [0, 0, 255]
            ax_img_another[3].imshow(cv2.addWeighted(img, 1, overlay, 0.5, 0))
            ax_img_another[3].title.set_text('Overlay')

            plt.axis('off')
            fig_img_another.savefig(os.path.join(plot_more, class_name + '_overlay_{}'.format(i)), dpi=100)
            plt.close()
