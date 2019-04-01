import h5py
import numpy as np
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt


class MSMT_view_masked(data.Dataset):
    def __init__(self, root, state='train', transform=None, require_views=True,
                 view_labeled=[1, 2, 3, 4, 5,]):
        super(MSMT_view_masked, self).__init__()
        self.root = root
        self.state = state
        self.transform = transform
        self.require_views = require_views
        self.num_classes = None
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False

        f = h5py.File(self.root, 'r')
        variables = list(f.items())
        # [0]: gallery_data
        # [1]: gallery_labels
        # [2]: gallery_views
        # [3]: probe_data
        # [4]: probe_labels
        # [5]: probe_views
        # [6]: train_data
        # [7]: train_labels
        # [8]: train_views

        if self.state == 'train':
            _, temp = variables[8]
            views = np.squeeze(temp.value)
            idx_mask = np.zeros(views.size, dtype=bool)
            for i in range(len(view_labeled)):
                idx_tmp = views == view_labeled[i]
                idx_mask[idx_tmp] = True
            self.views = views[idx_mask]
            _, temp = variables[6]
            self.data = np.transpose(temp.value, (0, 3, 2, 1))
            self.data = self.data[idx_mask, :, :, :]
            _, temp = variables[7]
            self.labels = np.squeeze(temp.value)
            self.num_classes = np.size(np.unique(self.labels))
            self.labels = self.labels[idx_mask]
        elif self.state == 'gallery':
            _, temp = variables[0]
            self.data = np.transpose(temp.value, (0, 3, 2, 1))
            _, temp = variables[1]
            self.labels = np.squeeze(temp.value)
            self.num_classes = np.size(np.unique(self.labels))
            _, temp = variables[2]
            self.views = np.squeeze(temp.value)
        elif self.state == 'probe':
            _, temp = variables[5]
            views = np.squeeze(temp.value)
            idx_mask = np.ones(views.size, dtype=bool)
            for i in range(len(view_labeled)):
                idx_tmp = views == view_labeled[i]
                idx_mask[idx_tmp] = False
            self.views = views[idx_mask]
            _, temp = variables[3]
            self.data = np.transpose(temp.value, (0, 3, 2, 1))
            self.data = self.data[idx_mask, :, :, :]
            _, temp = variables[4]
            self.labels = np.squeeze(temp.value)
            self.num_classes = np.size(np.unique(self.labels))
            self.labels = self.labels[idx_mask]
        else:
            assert False, 'Unknown state: {}\n'.format(self.state)

    def return_mean(self, axis=(0, 1, 2)):
        return np.mean(self.data, axis)

    def return_std(self, axis=(0, 1, 2)):
        return np.std(self.data, axis)

    def return_num_class(self):

        return self.num_classes

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        img, label, view = self.data[index], self.labels[index], self.views[index]

        img = Image.fromarray(img)

        if self.on_transform:
            img = self.transform(img)

        if self.require_views:
            return img, label, view
        else:
            return img, label



def main():
    MSMT_dataset = FullTraining('data/MSMT.mat')
    print(MSMT_dataset.__len__())
    img, label = MSMT_dataset[0]
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()