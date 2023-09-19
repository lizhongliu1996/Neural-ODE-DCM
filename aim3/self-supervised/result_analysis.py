import pickle
import numpy as np
import pandas as pd
import os

def analysis():
    # construct result dict
    result_path = 'result/finetune_seed1/'
    filename = 'fullmodel_ext_result.pkl'
    with open(os.path.join(result_path, filename), "rb") as f:
        result = pickle.load(f)
        
    test_result = result['test']
    test_best = np.argmax(test_result)
    
    
    print("test_best: ", test_best)
    
     
    pd.DataFrame(test_result).to_csv(f"result/{filename}.csv", index = None)
    
if __name__ == "__main__":
    analysis()
        
        
    
        
        

    
