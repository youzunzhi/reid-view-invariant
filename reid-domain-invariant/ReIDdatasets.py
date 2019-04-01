import h5py
import numpy as np
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt

class MSMT(data.Dataset):
    def __init__(self, root, transform=None, require_views=True, labeled=True):
        super(MSMT, self).__init__()
        self.root = root
        self.transform = transform
        self.require_views = require_views
        self.num_classes = None
        self.labeled = labeled
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False

        f = h5py.File(self.root, 'r')
        self.variables = list(f.items())
        # [0]: gallery_data
        # [1]: gallery_labels
        # [2]: gallery_views
        # [3]: probe_data
        # [4]: probe_labels
        # [5]: probe_views
        # [6]: train_data
        # [7]: train_labels
        # [8]: train_views

        self.data = None
        self.labels = None
        self.views = None


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
        return self.views.shape[0]

    def __getitem__(self, index):
        img, view = self.data[index], self.views[index]
        img = Image.fromarray(img)

        if self.on_transform:
            img = self.transform(img)
        if self.labeled:
            label = self.labels[index]
            if self.require_views:
                return img, label, view
            else:
                return img, label
        else:
            if self.require_views:
                return img, view
            else:
                return img



class MSMT_source_train(MSMT):
    def __init__(self, root, view_labeled, transform=None, require_views=True):
        super(MSMT_source_train, self).__init__(root, transform, require_views)

        _, temp = self.variables[8]
        views = np.squeeze(temp.value)
        idx_mask = np.zeros(views.size, dtype=bool)
        for i in range(len(view_labeled)):
            idx_tmp = views == view_labeled[i]
            idx_mask[idx_tmp] = True
        self.views = views[idx_mask]
        _, temp = self.variables[6]
        self.data = np.transpose(temp.value, (0, 3, 2, 1))
        self.data = self.data[idx_mask, :, :, :]
        _, temp = self.variables[7]
        self.labels = np.squeeze(temp.value)
        self.num_classes = np.size(np.unique(self.labels))
        self.labels = self.labels[idx_mask]


class MSMT_target_train(MSMT):
    def __init__(self, root, view_labeled, transform=None, require_views=True):
        super(MSMT_target_train, self).__init__(root, transform, require_views)

        _, temp = self.variables[8]
        views = np.squeeze(temp.value)
        idx_mask = np.ones(views.size, dtype=bool)
        for i in range(len(view_labeled)):
            idx_tmp = views == view_labeled[i]
            idx_mask[idx_tmp] = False
        self.views = views[idx_mask]
        _, temp = self.variables[6]
        self.data = np.transpose(temp.value, (0, 3, 2, 1))
        self.data = self.data[idx_mask, :, :, :]
        self.labeled = False

class MSMT_gallery(MSMT):
    def __init__(self, root, transform=None, require_views=True):
        super(MSMT_gallery, self).__init__(root, transform, require_views)

        _, temp = self.variables[0]
        self.data = np.transpose(temp.value, (0, 3, 2, 1))
        _, temp = self.variables[1]
        self.labels = np.squeeze(temp.value)
        self.num_classes = np.size(np.unique(self.labels))
        _, temp = self.variables[2]
        self.views = np.squeeze(temp.value)

class MSMT_probe(MSMT):
    def __init__(self, root, view_labeled, transform=None, require_views=True):
        super(MSMT_probe, self).__init__(root, transform, require_views, )

        _, temp = self.variables[5]
        views = np.squeeze(temp.value)
        idx_mask = np.ones(views.size, dtype=bool)
        for i in range(len(view_labeled)):
            idx_tmp = views == view_labeled[i]
            idx_mask[idx_tmp] = False
        self.views = views[idx_mask]
        _, temp = self.variables[3]
        self.data = np.transpose(temp.value, (0, 3, 2, 1))
        self.data = self.data[idx_mask, :, :, :]
        _, temp = self.variables[4]
        self.labels = np.squeeze(temp.value)
        self.num_classes = np.size(np.unique(self.labels))
        self.labels = self.labels[idx_mask]

def main():
    MSMT_dataset = MSMT_source_train('data/MSMT.mat')
    print(MSMT_dataset.__len__())
    img, label = MSMT_dataset[0]
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()