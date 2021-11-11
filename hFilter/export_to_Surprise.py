import preprocessing
import pandas as pd


def export(df):
    topics_set = set(preprocessing.get_topic_frequency_map_topics_as_list(df).keys())
    result_list = []
    max_val = 0
    for index, row in df.iterrows():
        repo = row['repo']
        repo_topics = set(row['topics'])
        missing_topic = topics_set - repo_topics
        for topic in repo_topics:
            result_list.append([repo, topic, 1])
        for topic in missing_topic:
            result_list.append([repo, topic, 0])
        max_val = max_val + 1
        #if max_val > 200: break
    df2 = pd.DataFrame(result_list, columns=['ProjectID', 'LibID', 'rating'])
    return df2

def evaluate(df):
    from surprise import Dataset
    from surprise import Reader
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[['ProjectID', 'LibID', 'rating']], reader)
    n_splits = 10
    from surprise.model_selection import KFold
    from collections import defaultdict
    kf = KFold(n_splits=n_splits)
    is_user_based = True
    neighborhood = 40
    cutoff = 5
    sim_funct = 'cosine'
    sim_settings = {'name': sim_funct,
                    'user_based': is_user_based
                    }
    from surprise import KNNWithMeans
    algo = KNNWithMeans(k=neighborhood, sim_options=sim_settings)
    threshold = 0.1
    k = 10
    print("FOLDS")
    for trainset, testset in kf.split(data):
        # print("Fitting the model by training data")
        algo.fit(trainset)
        # print("Predicting score for the test data")
        predictions = algo.test(testset)

        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        precision = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)

        print("average precision", precision)
        print("average recall", recall)

        f1_measure = (2 * precision * recall) / (recall + precision)
        print("average f1_measure", f1_measure)
