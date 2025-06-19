# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Iterator, Optional, Sized, List
import torch
from torch.utils.data import Sampler
import json

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class BalanceSampler(Sampler):
    r"""根据给定的概率（权重）从 ``[0,..,len(weights)-1]`` 中采样元素。
    允许对不同类别进行不同的采样率设置。

    参数：
        annotation_file (str): COCO格式的注释文件路径。
        shuffle (bool): 是否打乱数据集。默认为 True。
        seed (int, optional): 随机种子。如果为 None，则设置随机种子。默认为 None。
        target_class (int, optional): 如果指定该参数，则仅包含目标类别的数据会被视为有效数据进行采样。
    """

    def __init__(self,
                 dataset: Sized,
                 annotation_file: str,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 target_class: Optional[int] = None) -> None:
        rank, world_size = get_dist_info()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.target_class = target_class

        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        # 从 COCO 格式的注释文件中提取类别标签
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        image_annotations = annotations['annotations']
        self.class_labels_dict = {}
        for ann in image_annotations:
            image_id = ann['image_id']
            category_id = ann['category_id']
            if image_id not in self.class_labels_dict:
                self.class_labels_dict[image_id] = set()
            self.class_labels_dict[image_id].add(category_id)

        # 将所有图像的类别标签收集为列表，确保每张图像只记录一个类别
        self.class_labels = []
        self.image_ids = list(self.class_labels_dict.keys())

        # 对于每个图像，选择一个类别作为代表类别（可以使用第一个类别或其他策略）
        for labels in self.class_labels_dict.values():
            main_label = list(labels)[0]  # 选择第一个类别作为代表类别
            self.class_labels.append(main_label)

        self.size = len(self.class_labels)
        self.indices = list(self._indices_of_rank())

    def _indices_of_rank(self) -> List[int]:
        """为当前 rank 生成采样索引列表。"""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = []

        # 对于只同时包含类别 3 和 7 的图像，只采样 50%，并保证采样是随机的
        class_3_and_7_indices = [idx for idx in range(len(self.class_labels)) if
                                 self.class_labels_dict[self.image_ids[idx]] == {3, 7}]
        num_samples_3_and_7 = int(len(class_3_and_7_indices) * 0.1)
        if num_samples_3_and_7 > 0:
            sampled_indices_3_and_7 = torch.randperm(len(class_3_and_7_indices), generator=g).tolist()[
                                      :num_samples_3_and_7]
            indices.extend([class_3_and_7_indices[i] for i in sampled_indices_3_and_7])
        # 对于只同时包含类别 7 的图像，只采样 10%，并保证采样是随机的
        class_7_indices = [idx for idx in range(len(self.class_labels)) if
                           self.class_labels_dict[self.image_ids[idx]] == {7}]
        num_samples_7 = int(len(class_7_indices) * 1)
        if num_samples_7 > 0:
            num_samples_7 = torch.randperm(len(class_7_indices), generator=g).tolist()[
                            :num_samples_7]
            indices.extend([class_7_indices[i] for i in num_samples_7])
        # 对于其他图像，全部采样，排除只包含类别 7 的样本和只包含类别 3 和 7 的样本
        other_indices = [idx for idx in range(len(self.class_labels))
                         if not (self.class_labels_dict[self.image_ids[idx]] == {7} or self.class_labels_dict[
                self.image_ids[idx]] == {3, 7})]
        indices.extend(other_indices)

        # 打乱采样后的索引
        if self.shuffle:
            indices = torch.tensor(indices)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # 分布式场景下对采样的索引进行切片
        return list(itertools.islice(indices, self.rank, None, self.world_size))

    def __iter__(self) -> Iterator[int]:
        """迭代采样索引。"""
        return iter(self.indices)

    def __len__(self) -> int:
        """数据集的长度。"""
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch。

        当 :attr:`shuffle=True` 时，这确保所有副本在每个 epoch 中使用不同的随机顺序。

        参数：
            epoch (int): epoch 数。
        """
        self.epoch = epoch
        self.indices = list(self._indices_of_rank())
