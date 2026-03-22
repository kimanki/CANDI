from torch.utils.data import DataLoader

from datasets.build import build_dataset


def construct_loader(cfg, split):
    if split == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = cfg.TRAIN.SHUFFLE
        drop_last = cfg.TRAIN.DROP_LAST
    elif split == "val":
        batch_size = cfg.VAL.BATCH_SIZE
        shuffle = cfg.VAL.SHUFFLE
        drop_last = cfg.VAL.DROP_LAST
    elif split == "test":
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = cfg.TEST.SHUFFLE
        drop_last = cfg.TEST.DROP_LAST
    else:
        raise ValueError

    dataset = build_dataset(cfg, split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last
    )

    return loader


def get_train_dataloader(cfg):
    return construct_loader(cfg, "train")


def get_val_dataloader(cfg):
    return construct_loader(cfg, "val")


def get_test_dataloader(cfg):
    return construct_loader(cfg, "test")
