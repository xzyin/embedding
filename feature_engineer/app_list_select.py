#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.feature_selection import VarianceThreshold

def var():
    var=VarianceThreshold(threshold=1.0)
    data=var.fit_transform([[0,2,0,3],[0,1,4,3],[0,1,1,3]])
    print(data)
    return None

var()