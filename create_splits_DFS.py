import pdb
import os
import pandas as pd
from datasets.dataset_mtl_concat_lymph import Generic_WSI_MTL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= -1,
										help='fraction of labels (default: [1.0])')
parser.add_argument('--seed', type=int, default=1,
										help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
										help='number of splits (default: 10)')
parser.add_argument('--hold_out_test', action='store_true', default=False,
										help='fraction to hold out (default: 0)')
parser.add_argument('--split_code', type=str, default=None)
parser.add_argument('--split_rate', type=float, default=0.2)
parser.add_argument('--task', type=str, choices=['DFS'])

args = parser.parse_args()

    
if args.task == 'DFS':
    patient_path = './target/DFS_all.csv'
    df_patient = pd.read_csv(patient_path, index_col=0)
    label_cols = list(df_patient.columns[2:])
    time_dic = {}
    for i in df_patient['Disease Free (Months)']:
        time_dic[i] = i
    args.n_classes = 2
    dataset = Generic_WSI_MTL_Dataset(csv_path=patient_path,
                                      shuffle=False,
                                      seed=args.seed,
                                      print_info=True,
                                      label_dicts=[{1: 1, 0: 0},
                                                   time_dic],
                                      label_cols=label_cols,
                                      patient_strat=True)
else:
	raise NotImplementedError
    
num_slides_cls1 = np.array([len(cls_ids1) for cls_ids1 in dataset.patient_cls_ids])
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.floor(num_slides_cls * args.split_rate).astype(int)
test_num = np.floor(num_slides_cls * 0).astype(int)
train_num = np.floor(num_slides_cls * (1-args.split_rate)).astype(int)
print(val_num)
print(test_num)
print(train_num)

if __name__ == '__main__':
		if args.label_frac > 0:
			label_fracs = [args.label_frac]
		else:
			label_fracs = [1.0]

		if args.hold_out_test:
			custom_test_ids = dataset.sample_held_out(test_num=test_num)
		else:
			custom_test_ids = None
		for lf in label_fracs:
			if args.split_code is not None:
				split_dir = 'splits/'+ str(args.split_code) + '_{}'.format(int(lf * 100))
			else:
				split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
			
			dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf, custom_test_ids=custom_test_ids)

			os.makedirs(split_dir, exist_ok=True)
			for i in range(args.k):
				if dataset.split_gen is None:
					ids = []
					for split in ['train', 'val', 'test']:
						ids.append(dataset.get_split_from_df(pd.read_csv(os.path.join(split_dir, 'splits_{}.csv'.format(i))), split_key=split, return_ids_only=True))
					
					dataset.train_ids = ids[0]
					dataset.val_ids = ids[1]
					dataset.test_ids = ids[2]
				else:
					dataset.set_splits()

				descriptor_df = dataset.test_split_gen(return_descriptor=True)
				descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))
				
				splits = dataset.return_splits(from_id=True)
				save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
				save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
				



