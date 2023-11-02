import numpy as np
import yaml
from sacd import env_sac
from util import rowdata_process_util, json_util
from tqdm import tqdm


def sava_transition(env, pass_item_list):
    K = env.K
    s_all = env.observation.tolist()
    k = 0
    sars = []

    while True:
        pos_s = env.reset(s_all[k:k + K])
        pos_a = int(s_all[k + K:k + K + 1][0])
        pos_s_, r, done = env.step(pos_a, pass_item_list)

        pos_transition = np.hstack((pos_s, [pos_a, r], pos_s_))
        sars.append(pos_transition)

        if k + K + 1 == len(s_all):
            break
        else:
            k += 1
    return sars


def create_trans(dataset_name):
    path = "../config/sacd.yaml"

    with open(path, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    dataset = rowdata_process_util.Dataset('../dataset/' + dataset_name + '/ratings.dat')  # 读取数据 ratings.dat
    train_dict = dataset.splitData()
    item_id_list, item_num, max_item_id = dataset.getAllItem()
    item_hr_dict = json_util.load_dict('../dataset/' + dataset_name + '/click_through_rate.json')
    pop_dict = dataset.getPopular(train_dict)
    K = config['K']
    for each_user in tqdm(train_dict, total=len(train_dict)):
        pass_item_list = []
        for user_id, item_id, ratings, __ in dataset.data:
            if user_id == each_user and ratings == 0:
                pass_item_list.append(item_id)

        observation_data = train_dict[each_user]
        env = env_sac.Env(observation_data, list(set(item_id_list)), max_item_id, each_user, K, item_hr_dict, pop_dict)

        if env.n_observation <= K + 1:
            transition = [0]
            np.savetxt('../dataset/' + dataset_name + '/transition/' + str(each_user) + '_transition.csv', transition,
                       fmt='%i')
        else:
            transition = sava_transition(env, pass_item_list)
            fmt_str = ""
            for i in range(K):
                fmt_str += "%i|"
            fmt_str = fmt_str.rstrip('|') + ",%i,%f," + fmt_str.rstrip('|')
            np.savetxt('../dataset/' + dataset_name + '/transition/' + str(each_user) + '_transition.csv', transition,
                       fmt=fmt_str)


if __name__ == "__main__":
    create_trans('ml-1m')
