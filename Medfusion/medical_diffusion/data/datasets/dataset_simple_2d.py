
import torch.utils.data as data 
import torch 
from torch import nn
from pathlib import Path 
from torchvision import transforms as T
import pandas as pd
import numpy as np

from PIL import Image
from datetime import datetime

from medical_diffusion.data.augmentation.augmentations_2d import Normalize, ToTensor16bit

class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = 'tif', # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform = None,
        image_resize = None,
        augment_horizontal_flip = False,
        augment_vertical_flip = False, 
        image_crop = None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

        if transform is None: 
            self.transform = T.Compose([
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),
                T.ToTensor(),
                # T.Lambda(lambda x: torch.cat([x]*3) if x.shape[0]==1 else x),
                # ToTensor16bit(),
                # Normalize(), # [0, 1.0]
                # T.ConvertImageDtype(torch.float),
                T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        # img = Image.open(path_item) 
        img = self.load_item(path_item)
        return {'uid':rel_path_item.stem, 'source': self.transform(img)}
    
    def load_item(self, path_item):
        return Image.open(path_item).convert('RGB') 
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images 
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None 


class AIROGSDataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = pd.read_csv(self.path_root.parent/'train_labels.csv', index_col='challenge_id')
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        uid = self.labels.index[index]
        path_item = self.path_root/f'{uid}.jpg'
        img = self.load_item(path_item)
        str_2_int = {'NRG':0, 'RG':1} # RG = 3270, NRG = 98172 
        target = str_2_int[self.labels.loc[uid, 'class']]
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}
    
    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1/self.labels['class'].value_counts(normalize=True) # {'NRG': 1.03, 'RG': 31.02}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.iloc[index]['class']
            weights[index] = weight_per_class[target]
        return weights
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

class MSIvsMSS_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.2530835
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {'MSIMUT':0, 'MSS':1}
        target = str_2_int[path_item.parent.name] #
        return {'uid':uid, 'source': self.transform(img), 'target':target}


class MSIvsMSS_2_Dataset(SimpleDataset2D):
    # https://doi.org/10.5281/zenodo.3832231
    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = rel_path_item.stem
        str_2_int = {'MSIH':0, 'nonMSIH':1} # patients with MSI-H = MSIH; patients with MSI-L and MSS = NonMSIH)
        target = str_2_int[path_item.parent.name] 
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}


class CheXpert_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode = self.path_root.name
        labels = pd.read_csv(self.path_root.parent/f'{mode}.csv', index_col='Path')
        self.labels = labels.loc[labels['Frontal/Lateral'] == 'Frontal'].copy()
        self.labels.index = self.labels.index.str[20:]
        self.labels.loc[self.labels['Sex'] == 'Unknown', 'Sex'] = 'Female' # Affects 1 case, must be "female" to match stats in publication
        self.labels.fillna(2, inplace=True) # TODO: Find better solution, 
        str_2_int = {'Sex': {'Male':0, 'Female':1}, 'Frontal/Lateral':{'Frontal':0, 'Lateral':1}, 'AP/PA':{'AP':0, 'PA':1}}
        self.labels.replace(str_2_int, inplace=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rel_path_item = self.labels.index[index]
        path_item = self.path_root/rel_path_item
        img = self.load_item(path_item)
        uid = str(rel_path_item)
        target = torch.tensor(self.labels.loc[uid, 'Cardiomegaly']+1, dtype=torch.long)  # Note Labels are -1=uncertain, 0=negative, 1=positive, NA=not reported -> Map to [0, 2], NA=3
        return {'uid':uid, 'source': self.transform(img), 'target':target}

    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

class CheXpert_2_Dataset(SimpleDataset2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = pd.read_csv(self.path_root/'labels/cheXPert_label.csv', index_col=['Path', 'Image Index']) # Note: 1 and -1 (uncertain) cases count as positives (1), 0 and NA count as negatives (0)
        labels = labels.loc[labels['fold']=='train'].copy() 
        labels = labels.drop(labels='fold', axis=1)

        labels2 = pd.read_csv(self.path_root/'labels/train.csv', index_col='Path')
        labels2 = labels2.loc[labels2['Frontal/Lateral'] == 'Frontal'].copy()
        labels2 = labels2[['Cardiomegaly',]].copy()
        labels2[ (labels2 <0) | labels2.isna()] = 2 # 0 = Negative, 1 = Positive, 2 = Uncertain
        labels = labels.join(labels2['Cardiomegaly'], on=["Path",], rsuffix='_true')
        # labels = labels[labels['Cardiomegaly_true']!=2]

        self.labels = labels 
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path_index, image_index = self.labels.index[index]
        path_item = self.path_root/'data'/f'{image_index:06}.png'
        img = self.load_item(path_item)
        uid = image_index
        target = int(self.labels.loc[(path_index, image_index), 'Cardiomegaly'])
        # return {'uid':uid, 'source': self.transform(img), 'target':target}
        return {'source': self.transform(img), 'target':target}
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []
    
    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1/self.labels['Cardiomegaly'].value_counts(normalize=True)
        # weight_per_class = {2.0: 1.2, 1.0: 8.2, 0.0: 24.3}
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.loc[self.labels.index[index], 'Cardiomegaly']
            weights[index] = weight_per_class[target]
        return weights

# class FundusEyeDiseaseDataset(SimpleDataset2D):
#     """眼底疾病条件数据集"""
#     def __init__(self, csv_path, *args, **kwargs):
#         # 不需要传path_root，因为CSV中包含完整路径
#         super().__init__(path_root='/', *args, **kwargs)  # 设置一个dummy root
        
#         self.labels_df = pd.read_csv(csv_path)
        
#         # 我们关注的6种疾病
#         self.disease_columns = [
#             '青光眼', 
#             '糖尿病性视网膜病变', 
#             '年龄相关性黄斑变性', 
#             # '病理性近视',
#             '白内障',
#             '视网膜静脉阻塞',
#             '正常眼底'
#         ]
        
#         print(f"数据集加载完成，共 {len(self.labels_df)} 个样本")
#         for disease in self.disease_columns:
#             count = self.labels_df[disease].sum()
#             print(f"  {disease}: {count} 个样本")
    
#     def __len__(self):
#         return len(self.labels_df)
        
#     def __getitem__(self, index):
#         row = self.labels_df.iloc[index]
        
#         # 使用CSV中的完整路径
#         img_path = Path(row['img_path'])
        
#         # 检查文件是否存在
#         if not img_path.exists():
#             raise FileNotFoundError(f"图像文件不存在: {img_path}")
            
#         # 加载图像
#         img = self.load_item(img_path)
        
#         # 提取疾病的multi-hot向量
#         disease_vector = torch.tensor(
#             row[self.disease_columns].values.astype(np.float32), 
#             dtype=torch.float32
#         )

#         label_id = torch.argmax(disease_vector).long()
        
#         return {
#             'source': self.transform(img),
#             'target': label_id,  # 6维疾病向量
#             'img_path': str(img_path)  # 用于调试
#         }
    
#     @classmethod
#     def run_item_crawler(cls, path_root, extension, **kwargs):
#         """重写此方法，因为我们从CSV获取文件列表"""
#         return []  # 返回空列表，我们不使用crawler

class FundusEyeDiseaseDataset(SimpleDataset2D):
    """眼底疾病 + 眼别 + 图像质量 条件数据集"""
    def __init__(self, csv_path, *args, **kwargs):
        super().__init__(path_root='/', *args, **kwargs)
        self.labels_df = pd.read_csv(csv_path)

        # ✅ 去掉“正常眼底”，保留5个疾病
        self.disease_columns = [
            '青光眼',
            '糖尿病性视网膜病变',
            '年龄相关性黄斑变性',
            '白内障',
            '视网膜静脉阻塞'
        ]

        # ✅ 检查是否包含新列
        assert '左右眼' in self.labels_df.columns, "CSV中缺少列: 左右眼"

        print(f"数据集加载完成，共 {len(self.labels_df)} 个样本")
        for disease in self.disease_columns:
            print(f"  {disease}: {self.labels_df[disease].sum()} 个样本")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        row = self.labels_df.iloc[index]
        img_path = Path(row['img_path'])

        if not img_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {img_path}")

        img = self.load_item(img_path)

        # ✅ 疾病multi-hot向量 (float)
        disease_vec = torch.tensor(
            row[self.disease_columns].values.astype(np.float32),
            dtype=torch.float32
        )

        # ✅ 眼别 (0=右眼, 1=左眼)
        eye_side = torch.tensor(int(row['左右眼']), dtype=torch.long)

        return {
            'source': self.transform(img),
            'disease_vec': disease_vec,
            'eye_side': eye_side,
            'img_path': str(img_path)
        }

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return []

class SynFundusDataset(SimpleDataset2D):
    """
    SynFundus 数据集：
    - 由 CSV 索引控制，文件名为 {idx}.png
    - 初始化阶段：不再检查数据完整性
    - 运行阶段：若遇异常则记录日志并跳过
    """

    def __init__(self, csv_path, image_dir=None, *args, **kwargs):
        super().__init__(path_root=image_dir or "/", *args, **kwargs)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.labels_df = pd.read_csv(csv_path)

        self.label_columns = [
            'dr_grade', 'eye_side',
            'is_amd', 'is_aon', 'is_crp', 'is_dm', 'is_dme',
            'is_em', 'is_gc', 'is_htr', 'is_pm', 'is_rvo',
            'is_fundus', 'is_optic_disc_readable', 'is_retinal_region_readable'
        ]
        # self.label_columns = [
        #     '青光眼',
        #     '糖尿病性视网膜病变',
        #     '年龄相关性黄斑变性',
        #     '白内障',
        #     '视网膜静脉阻塞',
        #     '左右眼'
        # ]

        print(f"SynFundus 数据集加载完成: 共 {len(self.labels_df)} 个样本")

        # ✅ 日志目录
        self.log_dir = Path("/data1/lhy/medfusion-main/a-log")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.exception_log = self.log_dir / "exception.txt"

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        try:
            row = self.labels_df.iloc[index]

            if self.image_dir == None:
                img_path = Path(row['img_path'])
            else:
                img_path = self.image_dir / f"{row['idx']}.png"

            # 加载图像
            img = self.load_item(img_path)
            img_tensor = self.transform(img)

            # 打包标签
            label_tensor = torch.tensor(
                row[self.label_columns].values.astype(np.float32),
                dtype=torch.float32
            )

            return {
                'source': img_tensor,
                'labels': label_tensor,
                'img_path': str(img_path)
            }

        except Exception as e:
            # ✅ 记录异常日志并跳过样本
            with open(self.exception_log, "a") as f:
                f.write(f"[{datetime.now()}] Error at index {index}: {e}\n")
            # 返回下一个样本，避免 None 进入 DataLoader
            return self.__getitem__((index + 1) % len(self))

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """我们从 CSV 加载，不需要爬取"""
        return []


class FundusControlNetDataset(SimpleDataset2D):
    """Dataset that aligns images, conditioning labels and structural control maps.

    Control images (e.g. vessel masks or optic disc/cup masks) are expected to share the
    same filename stem as the fundus image.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        vessel_dir: str,
        disc_cup_dir: str | None = None,
        image_column: str = "img_path",
        label_columns=None,
        control_ext: str = "png",
        control_resize: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(path_root=image_dir, item_pointers=[], *args, **kwargs)
        self.image_dir = Path(image_dir)
        self.vessel_dir = Path(vessel_dir)
        self.disc_cup_dir = Path(disc_cup_dir) if disc_cup_dir is not None else None
        self.image_column = image_column
        self.control_ext = control_ext
        self.control_resize = control_resize or kwargs.get("image_resize", None)

        self.labels_df = pd.read_csv(csv_path)

        default_label_columns = [
            'dr_grade', 'eye_side',
            'is_amd', 'is_aon', 'is_crp', 'is_dm', 'is_dme',
            'is_em', 'is_gc', 'is_htr', 'is_pm', 'is_rvo',
            'is_fundus', 'is_optic_disc_readable', 'is_retinal_region_readable'
        ]
        self.label_columns = label_columns or default_label_columns

        self.control_transform = T.Compose([
            T.Resize(self.control_resize) if self.control_resize is not None else nn.Identity(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

        self.control_channels = (2 if self.disc_cup_dir is not None else 1) * 3

    def __len__(self):
        return len(self.labels_df)

    def _resolve_image_path(self, row):
        img_value = Path(row[self.image_column])
        return img_value if img_value.is_absolute() else (self.image_dir / img_value)

    def _load_control(self, stem: str):
        control_images = []
        vessel_path = self.vessel_dir / f"{stem}.{self.control_ext}"
        control_images.append(Image.open(vessel_path).convert('RGB'))

        if self.disc_cup_dir is not None:
            disc_cup_path = self.disc_cup_dir / f"{stem}.{self.control_ext}"
            control_images.append(Image.open(disc_cup_path).convert('RGB'))

        control_tensor = torch.cat([self.control_transform(img) for img in control_images], dim=0)
        return control_tensor

    def __getitem__(self, index):
        row = self.labels_df.iloc[index]
        img_path = self._resolve_image_path(row)
        img = self.load_item(img_path)

        stem = img_path.stem
        control_tensor = self._load_control(stem)

        label_tensor = torch.tensor(
            row[self.label_columns].values.astype(np.float32),
            dtype=torch.float32
        )

        return {
            'source': self.transform(img),
            'labels': label_tensor,
            'control': control_tensor,
            'img_path': str(img_path)
        }
