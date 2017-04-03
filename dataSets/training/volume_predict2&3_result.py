#coding=utf-8
'''
整合volume_predict2和volume_predict3结果
'''

import numpy as np
import pandas as pd

volume_predict2 = pd.read_csv("volume_predict2_result.csv")
volume_predict3 = pd.read_csv("volume_predict3_result.csv")

volume_predict3["volume"] = 0.3 * volume_predict3["volume"] + 0.7 * volume_predict2["volume"]

volume_predict3.to_csv("volume_predict2&3_result.csv")