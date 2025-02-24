from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


from src import environment


class Hw1Env(environment.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        r = np.random.rand()
        if r < 0.5:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "box", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.02, 0.005, 0.0001],
                                      density=4000, name="obj1")
        else:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "sphere", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.2, 0.005, 0.0001],
                                      density=4000, name="obj1")
        return scene

    def state(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return obj_pos, pixels

    def step(self, action_id):
        if action_id == 0:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 1:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 2:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 3:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})


def collect(idx, N):
    env = Hw1Env(render_mode="offscreen")
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    for i in range(N):
        env.reset()
        action_id = np.random.randint(4)
        _, img_before = env.state()
        imgs_before[i] = img_before
        env.step(action_id)
        obj_pos, img_after = env.state()
        positions[i] = torch.tensor(obj_pos)
        actions[i] = action_id
        imgs_after[i] = img_after
    torch.save(positions, f"HW1/data/positions/positions_{idx}.pt")
    torch.save(actions, f"HW1/data/actions/actions_{idx}.pt")
    torch.save(imgs_before, f"HW1/data/images/imgs_{idx}.pt")
    torch.save(imgs_after, f"HW1/data/images_after/imgs_after_{idx}.pt")


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.positions = []
        self.actions = []
        self.images = []
        self.images_after = []

        # load from multiple files
        for i in range(4):
            self.positions.append(torch.load(f"data/positions/positions_{i}.pt"))
            self.actions.append(torch.load(f"data/actions/actions_{i}.pt"))
            self.images.append(torch.load(f"data/images/imgs_{i}.pt"))
            self.images_after.append(torch.load(f"data/images_after/imgs_after_{i}.pt"))

        # concatenation
        self.positions = torch.cat(self.positions, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        self.images = torch.cat(self.images, dim=0)
        self.images_after = torch.cat(self.images_after, dim=0)

        # sanity check
        assert len(self.positions) == len(self.actions) == len(self.images) == len(self.images_after), "Data length mismatch"

        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((128, 128)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        # ])

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = self.positions[idx]
        action = self.actions[idx]
        image = self.images[idx].float() / 255.0
        image_after = self.images_after[idx].float() / 255.0
        return position, action, image, image_after


def load_dataset():
    full_dataset = Dataset()
    train_size = int(0.8 * len(full_dataset))
    validation_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                                    [train_size, validation_size,
                                                                                     test_size])

    dataLoader_training = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataLoader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)
    dataLoader_testing = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    return dataLoader_training, dataLoader_validation, dataLoader_testing



if __name__ == "__main__":

    # Data Collecting
    processes = []
    for i in range(4):
        p = Process(target=collect, args=(i, 250))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("data collection is done")


    # Loading collected data
    # dataloader_training, dataloader_validation, dataloader_testing = load_dataset()

    # ...







