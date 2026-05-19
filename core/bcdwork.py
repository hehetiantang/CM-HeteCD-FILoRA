# core/bcdwork.py

import os
import sys
import time
import random
import logging
import numpy as np
import paddle

from paddleseg.utils import logger

from .datasets import CDReader, HeteCDReader
from .cdmisc.train import train


def worker_init_fn(worker_id):
    np.random.seed(32767 + worker_id)
    random.seed(32767 + worker_id)


class Work(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args

        self.logger()
        self.dataloader()
        train(model, self.train_loader, self.val_loader, self.test_loader, self.args)

    def logger(self):
        data_root = getattr(self.args, "data_root", "/data/sjh/data")
        self.dataset_path = os.path.join(data_root, self.args.dataset)

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        time_str = time.strftime("%Y_%m_%d_%H", time.localtime())

        save_dataset_name = self.args.dataset.lower()
        self.args.save_dir = os.path.join(
            self.args.root,
            save_dataset_name,
            "{}_{}".format(self.args.model, time_str),
        )

        os.makedirs(self.args.save_dir, exist_ok=True)

        self.args.best_model_path = os.path.join(
            self.args.save_dir,
            "{}_best.pdparams".format(self.args.model),
        )

        self.args.metric_path = os.path.join(
            self.args.save_dir,
            "{}_metrics.csv".format(self.args.model),
        )

        log_path = os.path.join(
            self.args.save_dir,
            "train_{}.log".format(self.args.model),
        )

        self.args.log_path = log_path

        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format="%(asctime)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        # 关键修改：把 PaddleSeg logger 挂到 args 上
        # 这样 val.py / train.py / predict.py 里使用 args.logger 就不会报错
        self.args.logger = logger

        logger.info("\nConfig:")
        for k, v in sorted(vars(self.args).items()):
            if k == "logger":
                logger.info("'{}': {}".format(k, "paddleseg_logger"))
            else:
                logger.info("'{}': {}".format(k, repr(v)))

        logger.info("Model {}, Datasets {}".format(self.args.model, self.args.dataset))
        logger.info("Dataset path {}".format(self.dataset_path))
        logger.info("lr {}, batch_size {}".format(self.args.lr, self.args.batch_size))
        logger.info(
            "log save at {}, metric save at {}, weight save at {}".format(
                log_path, self.args.metric_path, self.args.best_model_path
            )
        )

        print(
            "log save at {}, metric save at {}, weight save at {}".format(
                log_path, self.args.metric_path, self.args.best_model_path
            )
        )

    def dataloader(self, datasetlist=["train", "val", "test"]):
        input_mode = getattr(self.args, "input_mode", "hetecd")

        if input_mode == "hetecd":
            Reader = HeteCDReader
        else:
            Reader = CDReader

        train_data = Reader(
            self.dataset_path,
            "train",
            img_size=self.args.img_size,
        )

        val_data = Reader(
            self.dataset_path,
            "val",
            img_size=self.args.img_size,
        )

        test_dir = os.path.join(self.dataset_path, "test")
        if os.path.exists(test_dir):
            test_split = "test"
        else:
            test_split = "val"

        test_data = Reader(
            self.dataset_path,
            test_split,
            img_size=self.args.img_size,
        )

        self.args.traindata_num = train_data.__len__()
        self.args.val_num = val_data.__len__()
        self.args.test_num = test_data.__len__()

        batch_sampler = paddle.io.BatchSampler(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.train_loader = paddle.io.DataLoader(
            train_data,
            batch_sampler=batch_sampler,
            num_workers=self.args.num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn,
        )

        val_batch_sampler = paddle.io.BatchSampler(
            val_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.val_loader = paddle.io.DataLoader(
            val_data,
            batch_sampler=val_batch_sampler,
            num_workers=self.args.num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn,
        )

        test_batch_sampler = paddle.io.BatchSampler(
            test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.test_loader = paddle.io.DataLoader(
            test_data,
            batch_sampler=test_batch_sampler,
            num_workers=self.args.num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn,
        )
