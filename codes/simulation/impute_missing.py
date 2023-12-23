#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:42:08 2023

@author: yumsun
"""

import numpy as np


def impute_mean(x, missing_id):
    missing_id_rev = ((missing_id - 1)*(-1)).astype(int)
    masked_mean = np.ma.array(x, mask=missing_id_rev).mean(axis=0)
    impute_array = np.where(missing_id_rev.astype(bool),masked_mean,
                            x)
    return impute_array


if __name__ == '__main__':
    print('main')


