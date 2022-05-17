import os
import json
import numpy as np

def evaluate(suffix:str):
    class_index = {
        0: 'bus',
        1: 'cargo_truck',
        2: 'crane',
        3: 'tractor',
        4: 'van',
        5: 'excavator',
    }

    prediction_results = {
        'bus': 0,
        'cargo_truck': 0,
        'crane': 0,
        'tractor': 0,
        'van': 0,
        'excavator': 0,
    }

    result_path = os.path.join('..', 'result', f'predictions{suffix}.json')

    with open(result_path, 'r') as f:
        results = json.load(f)
    
    for pred in results.values():
        pred_arr = np.asarray(pred)
        pred_argmax = pred_arr.argmax()
        prediction_results[class_index[pred_argmax]] += 1
    
    print(f'Prediction results with suffix: {suffix}')
    print(prediction_results)
    accuracy_string = f'Accuracy: {prediction_results["excavator"]/sum([v for v in prediction_results.values()])*100}%'
    print(accuracy_string)

if __name__=='__main__':
    model_suffixes = [path.split('.')[0].split('model')[-1] for path in os.listdir(os.path.join('pretrained')) if path.endswith('.h5')]
    for suffix in model_suffixes:
        evaluate(suffix)
        print('\n')