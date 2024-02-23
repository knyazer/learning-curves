import lcdb
from lcdb import get_all_curves
import pandas as pd
import numpy as np
import torch
from typing import Callable, List, Any, Dict


class LCDB:
    data: pd.DataFrame
    def __init__(self, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = get_all_curves()

    def isin(self, row_name=None, values=None):
        if row_name is None or values is None:
            raise ValueError("Both row_name and values must be provided.")
        return LCDB(self.data[self.data[row_name].isin(values)])

    def to_curve_list(self):
        grouped = self.data.sort_values('size_train').groupby('size_train')


        dirty_anchors = grouped['size_train'].groups.keys()

        min_exp = 4
        max_exp_int = 50
        curves_anchor_first = {}

        for exp in range(0, max_exp_int + 1):
            anchor = int(np.round(2**(min_exp + 0.5 * exp)))
            if anchor not in dirty_anchors:
                continue

            group = grouped.get_group(anchor)
            curves_anchor_first[anchor] = group.values

        num_curves = max(len(group) for group in curves_anchor_first.values())
        num_anchors = len(curves_anchor_first)

        train_array = np.full((num_curves, num_anchors), np.nan)
        valid_array = np.full((num_curves, num_anchors), np.nan)
        test_array = np.full((num_curves, num_anchors), np.nan)
        meta_features = {}

        anchor_indices = {anchor: idx for idx, anchor in
                          enumerate(sorted(curves_anchor_first.keys()))}

        # Iterate and populate the arrays
        curve_id_registry = {}
        for anchor, group in curves_anchor_first.items():
            anchor_idx = anchor_indices[anchor]
            for row in group:
                curve_id = f'{row[0]}_{row[1]}_{row[5]}'
                if curve_id not in curve_id_registry.keys():
                    curve_id_registry[curve_id] = len(curve_id_registry)
                curve_index = curve_id_registry[curve_id]

                train_array[curve_index, anchor_idx] = row[-3]  # Assuming row[0] is train
                valid_array[curve_index, anchor_idx] = row[-2]  # Assuming row[1] is valid
                test_array[curve_index, anchor_idx] = row[-1]
                meta_features[curve_index] = row[:-3]

        return anchors, values_train, values_valid, values_test


if __name__ == '__main__':
    dataset = LCDB()

    dataset = dataset.isin('openmlid', list(range(100)))
    dataset = dataset.isin('learner', ['SVC_linear'])
    curves = dataset.to_curve_list()

    print(curves)


