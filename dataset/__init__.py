from .vehicleid import VehicleID 
from .veri776 import VeRi776 
from .veri_wild import VeRi_Wild 
from .dataloader import get_dataloader
from .preprocessor import Preprocessor
from .market1501 import Market1501
DATASET_ID_NUM = {'Market1501':751, 'VeRi776':576 , 'VehicleID':13164, 'VeRi_Wild':30671}

__all__ = ['Market1501','VehicleID', 'VeRi776', 'VeRi_Wild', 'get_dataloader', 'DATASET_ID_NUM', 'Preprocessor']
