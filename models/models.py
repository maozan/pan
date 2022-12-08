from .HyperTransformer import *
from .Panformer import *
from .GPPNN import *
from .Pannet import *
from .HyperDSNet import *

MODELS = {  "HSIT": HyperTransformer,
            "HSIT_PRE": HyperTransformerPre,
            "Panformer": Panformer,
            "GPPNN": gppnn, 
            "HyperDSNet": HyperDSNet,
            "PanNet": PanNet,
            }