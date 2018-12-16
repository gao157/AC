import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns 
#import lightgbm as lgb
#from catboost import Pool, CatBoostClassifier
import itertools
import pickle, gzip
import glob
#from sklearn.preprocessing import StandardScaler
#from tsfresh.feature_extraction import extract_features
#from sklearn.cluster import MeanShift, estimate_bandwidth
import multiprocessing 
import time
#import cesium.featurize
import scipy

np.warnings.filterwarnings('ignore')

def featurize_flux_sq(train):
	"""add flux_ratio_sq and flux_by_flux_ratio_sq"""
	train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
	train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']
	return train 

def featurize_agg(train):
	"""aggregate training data and return summary statistics"""
	aggs = {
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

	agg_df = train.groupby('object_id').agg(aggs)
	new_cols = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
	agg_df.columns = new_cols
	return agg_df 

def featurize_flux(train):
	"""add some random stuff"""
	train['flux_diff'] = train['flux_max'] - train['flux_min']
	train['flux_dif2'] =  (train['flux_max'] - train['flux_min']) / train['flux_mean']
	train['flux_w_mean'] = train['flux_by_flux_ratio_sq_sum'] / train['flux_ratio_sq_sum']
	train['flux_dif3'] = (train['flux_max'] - train['flux_min']) / train['flux_w_mean']
	return train 

def featurize_ts(train):
	"""add fft"""
	fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},{'coeff': 1, 'attr': 'abs'}],'kurtosis' : None, 'skewness' : None}
	df_features = extract_features(train, column_id='object_id', column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=4)
	return df_features

def featurize_ts_detected(train):
	"""add detected mjd diff"""
	df_features = train.loc[train['detected'] == 1, ['object_id', 'mjd']].groupby('object_id').agg({'mjd': ['min', 'max']})
	df_features.columns = ['mjd_min', 'mjd_max']
	df_features['mjd_diff_detected'] = df_features['mjd_max'] - df_features['mjd_min']
	df_features = df_features.drop(columns = ['mjd_max', 'mjd_min'])
	return df_features

def multi_weighted_logloss(y_true, y_preds):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


def lgb_multi_weighted_logloss(y_true, y_preds):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False

def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain', y='feature', data=importances_.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
    plt.savefig('importances.png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def create_mjd_cluster(x):
	array_ = np.array(x['mjd'])
	array_ = np.reshape(array_, (-1,1))
	ms = MeanShift(bandwidth=None, bin_seeding=True)
	ms.fit(array_)
	labels = ms.labels_
	x['mjd_cluster'] = labels
	return x

def featurize_mjd_cluster(df):
	new_df = df.groupby('object_id').apply(lambda x: create_mjd_cluster(x))
	return new_df

def featurize_mjd_cluster_w_multiprocessing(train, n_cores=8):
	data_split_list = np.array_split(train, n_cores)
	remain_df = None
	adjusted_data_split_list = []
	for df in data_split_list:
		unique_ids = np.unique(df['object_id'])
		new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
		if remain_df is None:
			df = df.loc[df['object_id'].isin(unique_ids[:-1])]
		else:
			df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
		remain_df = new_remain_df
		adjusted_data_split_list.append(df)

	pool = multiprocessing.Pool(n_cores)
	final_df = pd.concat(pool.map(featurize_mjd_cluster, adjusted_data_split_list))
	pool.close()
	pool.join()
	return final_df


def generate_list(train, col):
	groups = train.groupby(['object_id', 'passband']).apply(lambda block: block[col].values).reset_index().rename(columns={0: 'seq'})
	list_of_arrays = groups.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
	return list_of_arrays

def featurize_cesium(train):
	times_list = generate_list(train, 'mjd')
	flux_list = generate_list(train, 'flux')
	feats = cesium.featurize.featurize_time_series(
		times = times_list,
		values = flux_list,
		features_to_use = ['freq1_freq'],
		scheduler = None)
	return feats

def featurize_cesium_w_multiprocessing(train, n_cores=8):
	data_split_list = np.array_split(train, n_cores)
	remain_df = None
	adjusted_data_split_list = []
	counter = 1
	for df in data_split_list:
		unique_ids = np.unique(df['object_id'])
		new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
		if remain_df is None:
			df = df.loc[df['object_id'].isin(unique_ids[:-1])]
		else:
			if counter == n_cores:
				df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids)]], axis=0)
			else:
				df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
		remain_df = new_remain_df
		adjusted_data_split_list.append(df)
		counter += 1

	pool = multiprocessing.Pool(n_cores)
	final_df = pd.concat(pool.map(featurize_cesium, adjusted_data_split_list))
	pool.close()
	pool.join()
	final_df.columns = [str(tuple_[0]) + '_' + str(tuple_[1]) for tuple_ in final_df.columns]
	final_df['freq_var'] = final_df.var(axis=1)
	final_df.index = np.unique(train['object_id'])
	return final_df

def featurize_burst_events(train):
	groups = train.groupby(['object_id', 'passband'])
	start_time = time.time()
	cols = ['object_id', 'passband', 'initial_burst_rate','inital_decline_rate']
	det_df = pd.DataFrame(columns=cols)
	for i, (name, group) in enumerate(groups):
		group = group.reset_index()
		detected_percent = len(group[group['detected'] == 1]) / len(group)
		try:
			max_flux_idx = group[group['detected'] == 1].flux.idxmax()
		except ValueError:
			max_flux_idx = group.flux.idxmax()

		max_flux = group.loc[max_flux_idx].flux

		if max_flux_idx + 1 == len(group):
			inital_decline_rate = np.nan
		else:
			max_flux_1 = group.loc[max_flux_idx+1].flux
			inital_decline_rate = (max_flux - max_flux_1) / max_flux / (group.loc[max_flux_idx+1].mjd - group.loc[max_flux_idx].mjd)

		if max_flux_idx == 0:
			initial_burst_rate = np.nan  
		else:
			max_flux_1111 = group.loc[max_flux_idx-1].flux
			initial_burst_rate = (max_flux - max_flux_1111) / max_flux / (group.loc[max_flux_idx].mjd - group.loc[max_flux_idx-1].mjd)

		temp_df = pd.DataFrame([[name[0], name[1], initial_burst_rate,inital_decline_rate]], columns = cols)
		det_df = det_df.append(temp_df)

	det_df = det_df.set_index(['object_id', 'passband']).unstack()
	cols_name = [col_tuple[0] + '_' + str(col_tuple[1]) for col_tuple in det_df.columns]
	det_df.columns = cols_name
	return det_df

def featurize(train):
	"""calculate all features"""
	df_ts = featurize_ts(train)
	df_mjd = featurize_ts_detected(train)
	df_time = df_ts.merge(df_mjd, left_index = True, right_index = True)

	train = featurize_flux_sq(train)
	agg_train = featurize_agg(train)
	agg_train = featurize_flux(agg_train)

	agg_train = agg_train.merge(df_time, left_index = True, right_index = True)
	print('calculating cesium features...')
	df_cesium = featurize_cesium_w_multiprocessing(train)
	agg_train = agg_train.merge(df_cesium, left_index = True, right_index = True)
	print('calculating burst...')
	df_burst = featurize_burst_events(train)
	agg_train = agg_train.merge(df_burst, left_index = True, right_index = True)
	return agg_train

def calculate_slope(df):
	"""calculate the slope of flux from selected df"""
	x, y = df['mjd'], df['flux']
	if len(x) > 0:
		y_normalized = (y - y.mean()) / y.std()
		reg = scipy.stats.linregress(x, y_normalized)
		slope = reg.slope
	else:
		slope = np.nan
	return slope

train = pd.read_csv('training_set.csv', nrows=100000)

groups = train.groupby(['object_id', 'passband'])
for name, group in groups:
	print('group_name:', name)
	group = group.reset_index()
	detected_percent = len(group[group['detected'] == 1]) / len(group)
	if len(group[group['detected'] == 1]) > 1:

		det_df = group[group.detected == 1]
		first_det_idx, last_det_idx = det_df.index[0], det_df.index[-1]
		first_before_det_idx, last_after_det_idx = first_det_idx - 1, last_det_idx + 1
		max_flux_idx = det_df.flux.idxmax()

		# detected only decresing trend
		if max_flux_idx == first_det_idx:
			overall_decline_slope = calculate_slope(det_df)
			final_decline_slope = calculate_slope(group[group.mjd > group.loc[last_det_idx].mjd - 50])
			decline_ratio = final_decline_slope / overall_decline_slope

			if last_after_det_idx == len(group):
				final_decline_rate = np.nan
			else:
				if group.loc[last_after_det_idx].mjd - group.loc[last_det_idx].mjd < 50:
					final_decline_rate = (group.loc[last_after_det_idx].flux - group.loc[last_det_idx].flux) \
					/ group.loc[last_det_idx].flux / (group.loc[last_after_det_idx].mjd - group.loc[last_det_idx].mjd)
				else:
					final_decline_rate = np.nan 

		#detected only increasing trend
		elif max_flux_idx == last_det_idx:
			if first_before_det_idx == -1:
				initial_burst_rate = np.nan
			else:
				initial_burst_rate = (group.loc[max_flux_idx].flux - group.loc[first_before_det_idx].flux) / group.loc[first_before_det_idx].flux \
										/ (group.loc[max_flux_idx].mjd - group.loc[last_det_idx])

			overall_burst_slope = calculate_slope(det_df)
			initial_burst_slope = calculate_slope(group[group.mjd < group.loc[first_det_idx].mjd + 50])

		elif max_flux_idx > first_det_idx and max_flux_idx < last_det_idx:
			continue




			

'''

train = pd.read_csv('training_set.csv')
agg_train = pd.read_csv('agg_train.csv')
meta_train = pd.read_csv('training_set_metadata.csv')
full_train = agg_train.reset_index().merge(right=meta_train,how='outer',on='object_id')

if 'target' in full_train:
    y = full_train['target']
    del full_train['target']


classes = sorted(y.unique())
class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

oof_df = full_train['object_id']

if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'], full_train['distmod'], full_train['hostgal_specz']
    del full_train['ra'], full_train['decl'], full_train['gal_l'],full_train['gal_b'],full_train['ddf']

train_mean = full_train.mean(axis=0)
full_train.fillna(0, inplace=True)

w = y.value_counts()
weights = {i : np.sum(w) / w[i] for i in w.index}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
clfs = []
importances = pd.DataFrame()
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'learning_rate': 0.03,
    'subsample': .9,
    'colsample_bytree': 0.5,
    'reg_alpha': .01,
    'reg_lambda': .01,
    'min_split_gain': 0.01,
    'min_child_weight': 10,
    'n_estimators': 1000,
    'silent': -1,
    'verbose': -1,
    'max_depth': 3
}

w = y.value_counts()
weights = {i : np.sum(w) / w[i] for i in w.index}

oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
    val_x, val_y = full_train.iloc[val_], y.iloc[val_]

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        eval_metric=lgb_multi_weighted_logloss,
        verbose=100,
        early_stopping_rounds=50,
        sample_weight=trn_y.map(weights)
    )
    oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
    print(multi_weighted_logloss(val_y, oof_preds[val_, :]))

    imp_df = pd.DataFrame()
    imp_df['feature'] = full_train.columns
    imp_df['gain'] = clf.feature_importances_
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    clfs.append(clf)

print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y, y_preds=oof_preds))


save_importances(importances_=importances)
    
unique_y = np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i
        
y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])

# Compute confusion matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds,axis=-1))
np.set_printoptions(precision=2)
sample_sub = pd.read_csv('sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
del sample_sub;gc.collect()

# Plot non-normalized confusion matrix
plt.figure(figsize=(12,12))
plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                      title='Confusion matrix')
'''