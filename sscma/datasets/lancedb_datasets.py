from typing import List, Sequence
import pyarrow as pa
import numpy as np
from tqdm.std import tqdm
import os
import os.path as osp
from pycocotools.coco import COCO
import lance
from mmengine.dataset.base_dataset import force_full_init
import io
from PIL import Image
from sscma.datasets.base_dataset import BaseDataset


def process_images_detect(images_folder, schema, ann_file):
    coco = COCO(ann_file)
    images = coco.loadImgs(coco.getImgIds())
    images2id = {}
    for im_ann in images:
        images2id[im_ann["file_name"]] = im_ann["id"]
    image2ann = coco.imgToAnns

    for image_id, img_info in tqdm(coco.imgs.items()):
        image_file = img_info["file_name"]

        bboxes = []
        catids = []
        for ann in image2ann[image_id]:
            bboxes.append(ann["bbox"])
            catids.append(ann["category_id"])
        if len(bboxes) == 0:
            continue

        with open(osp.join(images_folder, image_file), "rb") as f:
            im = f.read()

        image_array = pa.array([im], type=pa.binary())
        filename_array = pa.array([str(image_file)], type=pa.string())
        bboxes_array = pa.array(
            [np.asarray(bboxes, dtype=np.float32).tobytes()], type=pa.binary()
        )
        catid_array = pa.array(
            [np.asarray(catids, dtype=np.int16).tobytes()], type=pa.binary()
        )

        # Yield RecordBatch for each image
        yield pa.RecordBatch.from_arrays(
            [image_array, filename_array, bboxes_array, catid_array],
            schema=schema,
        )


def process_images_cls(data_folder, schema):
    folders = [osp.join(data_folder, f) for f in os.listdir(data_folder)]

    for label, image_dir in enumerate(folders):
        image_files = [osp.join(image_dir, f) for f in os.listdir(image_dir)]
        for image_file in image_files:
            with open(image_file, "rb") as f:
                im = f.read()
            # im = cv2.imread(image_file)
            image_array = pa.array([im])
            filename_array = pa.array([str(image_file)], type=pa.string())
            labels = pa.array([label], type=pa.int8())

            # Yield RecordBatch for each image
            yield pa.RecordBatch.from_arrays(
                [image_array, filename_array, labels],
                schema=schema,
            )


# Function to write PyArrow Table to Lance dataset
def write_to_lance(lance_file, data_folder, ann_file=None):

    if ann_file != None:
        schema = pa.schema(
            [
                pa.field("image", pa.binary()),
                pa.field("filename", pa.string()),
                pa.field("bbox", pa.binary()),
                pa.field("catid", pa.binary()),
            ]
        )
        if osp.exists(lance_file):
            return schema

        batches = process_images_detect(data_folder, schema, ann_file)
    else:
        schema = pa.schema(
            [
                # pa.field("image", pa.list_(pa.list_(pa.list_(pa.int64())))),
                pa.field("image", pa.binary()),
                pa.field("filename", pa.string()),
                pa.field("label", pa.int8()),
            ]
        )
        if osp.exists(lance_file):
            return schema
        batches = process_images_cls(data_folder, schema)

    reader = pa.RecordBatchReader.from_batches(schema, batches)
    lance.write_dataset(
        reader,
        lance_file,
        schema,
    )
    return schema


class LanceDataset(BaseDataset):
    def __init__(
        self,
        ann_file: str = None,
        metainfo: dict = None,
        data_root: str = "",
        data_prefix: str = "",
        filter_cfg: dict = None,
        indices: int = None,
        serialize_data: bool = True,
        pipeline: Sequence = ...,
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
        classes: str = None,
    ):

        self.lance_file = osp.join(data_root, ".val" if test_mode else ".train")
        data_folder = osp.join(data_root, data_prefix.get("img_path", ""))
        if ann_file != None and not osp.isabs(ann_file):
            ann_file = osp.join(data_root, ann_file)

        self.cache_info = write_to_lance(
            self.lance_file, data_folder, ann_file=ann_file
        ).names

        if "image" in self.cache_info:
            self.cache_info.remove("image")

        self.ds = lance.dataset(self.lance_file)

        super().__init__(
            ann_file,
            metainfo,
            data_root,
            data_prefix,
            filter_cfg,
            indices,
            serialize_data,
            pipeline,
            test_mode,
            lazy_init,
            max_refetch,
            classes,
        )

    def __len__(self):
        return self.ds.count_rows()

    def load_data_list(self) -> List[dict]:
        anns = []
        for i in range(len(self)):
            data = self.ds.take([i], columns=self.cache_info).to_pydict()
            anns.append(data)

        return anns

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        data: dict = self.ds.take([idx]).to_pydict()
        if data.get("bbox", False):
            print(data["bbox"][0])
            # if data["bbox"][0]==None:
            data["catid"] = np.frombuffer(data["catid"][0], dtype=np.int16)
            data["bbox"] = np.frombuffer(data["bbox"][0], dtype=np.float32).reshape(
                -1, 4
            )
        data["image"] = Image.open(io.BytesIO(data["image"][0]))
        return data
