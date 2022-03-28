from collections import namedtuple

StockModel = namedtuple('StockModel', 'symbols network date epochs loss')
StockModelV2 = namedtuple('StockModelV2', 'symbols network date epochs null_model_loss loss')
