
import pandas as pd
from wildfiredb import WildFireData1, WildFireData4

def check_balance(dataset):
    print(f"Checking class balance for {dataset.DATASET_NAME}...")
    wf4 = dataset # WildFireData1()
    dfs = wf4.generate_dataframes()
    train_df, valid_df, test_df = dfs[0], dfs[1], dfs[2]

    print(f"Train set size: {len(train_df)}")
    print(train_df['Class'].value_counts(normalize=True))
    
    print(f"Valid set size: {len(valid_df)}")
    print(valid_df['Class'].value_counts(normalize=True))
    
    print(f"Test set size: {len(test_df)}")
    print(test_df['Class'].value_counts(normalize=True))
    
def check_combines_composition(datasets):
    combined_train_dfs = []
    combined_valid_dfs = []
    combined_test_dfs = []
    
    for dataset in datasets:
        dfs = dataset.generate_dataframes()
        combined_train_dfs.append(dfs[0])
        combined_valid_dfs.append(dfs[1])
        combined_test_dfs.append(dfs[2])
    
    combined_train_df = pd.concat(combined_train_dfs, ignore_index=True)
    combined_valid_df = pd.concat(combined_valid_dfs, ignore_index=True)
    combined_test_df = pd.concat(combined_test_dfs, ignore_index=True)
    
    print("Combined Train set class distribution:")
    print(combined_train_df['Class'].value_counts(normalize=True))
    
    print("Combined Valid set class distribution:")
    print(combined_valid_df['Class'].value_counts(normalize=True))
    
    print("Combined Test set class distribution:")
    print(combined_test_df['Class'].value_counts(normalize=True))

if __name__ == "__main__":
    # check_balance(WildFireData1())
    # check_balance(WildFireData4())
    # check_balance(WildFireData4())

    check_combines_composition([WildFireData1(), WildFireData4()])