"""
Simple script that checks if a checkpoint is corrupted with any inf/NaN values. Run like this:
    python inspect_checkpoint.py model.12345
"""
import numpy as np
import sys
import tensorflow as tf


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Usage: python inspect_checkpoint.py <file_name>\nNote: Do not include ' +
                        'the .data .index or .meta part of the model checkpoint in filename.')

    filename = sys.argv[1]
    reader = tf.train.NewCheckpointReader(filename)
    var_to_shape_map = reader.get_variable_to_shape_map()
    finite = []
    all_inf_nan = []
    some_inf_nan = []
    for key in sorted(var_to_shape_map.keys()):
        tensor = reader.get_tensor(key)
        if np.all(np.isfinite(tensor)):
            finite.append(key)
        else:
            if np.any(np.isfinite(tensor)):
                some_inf_nan.append(key)
            else:
                all_inf_nan.append(key)

    print('\nFINITE VARIABLES:')
    for key in finite:
        print(key)

    print('\nVARIABLES THAT ARE ALL INF/NAN:')
    for key in all_inf_nan:
        print(key)

    print('\nVARIABLES THAT CONTAIN SOME FINITE, SOME INF/NAN VALUES:')
    for key in some_inf_nan:
        print(key)

    if all_inf_nan or some_inf_nan:
        print('CHECK FAILED: checkpoint contains some inf/NaN values')
    else:
        print('CHECK PASSED: checkpoint contains no inf/NaN values')
