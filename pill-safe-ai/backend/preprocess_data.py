import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_optimized_loader(root_path, batch_size=256):
    """
    128GB RAM을 활용하여 데이터를 메모리에 미리 올리거나 
    병렬로 고속 로딩하는 전처리 로더입니다.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15), # 촬영 이미지의 각도 변화 대응
    ])