# File: data_provider/data_factory.py

# Only import the necessary Dataset class and DataLoader
from data_provider.data_loader import Dataset_Simplified_Finance
from torch.utils.data import DataLoader

def data_provider(args, flag):
    # flag: 'train', 'val', 'test', or 'pred'
    # timeenc = 0 if args.embed != 'timeF' else 1 # Use time features if embedding type is 'timeF'

    # Determine shuffle, drop_last, batch_size based on flag
    if flag == 'test' or flag == 'pred':
        shuffle_flag = False
        drop_last = False # Keep all data for testing/prediction
        batch_size = args.batch_size # Or set to 1 if needed for prediction logic later
    else: # 'train' or 'val'
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    # Instantiate your simplified dataset
    data_set = Dataset_Simplified_Finance(
        root_path=args.root_path,
        data_path=args.data_path, # e.g., 'my_data.csv'
        technical_indicators=args.technical_indicators,
        rolling_window=args.rolling_window,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features = args.features,
        scale=True,               # Or args.scale if you add it as an argument
    )

    print(f"{flag} set size: {len(data_set)}") # Print dataset size for verification

    # Create DataLoader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    # Return both the dataset object (for inverse_transform) and the loader
    return data_set, data_loader