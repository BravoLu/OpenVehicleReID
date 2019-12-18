from .vehicleid import VehicleID 
from .veri776 import VeRi776 
from .veri_wild import VeRi_Wild 
from .dataloader import get_dataloader
from .preprocessor import Preprocessor
DATASET_ID_NUM = {'VeRi776':576 , 'VehicleID':13164, 'VeRi_Wild':1000}

__all__ = ['VehicleID', 'VeRi776', 'VeRi_Wild', 'get_dataloader', 'DATASET_ID_NUM', 'Preprocessor']
