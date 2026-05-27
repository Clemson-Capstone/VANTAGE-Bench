from ..image_base import ImageBaseDataset


class VANTAGE_2DDetectionDataset(ImageBaseDataset):
    TYPE = "VQA"
    MODALITY = "IMAGE"

    @classmethod
    def supported_datasets(cls):
        return ['VANTAGE_2DDetection']
