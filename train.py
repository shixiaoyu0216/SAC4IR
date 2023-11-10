import copy
import csv
import os
from datetime import datetime

import yaml

from sacd import env_sac
from sacd.agent.sacd import SacdAgent
from util import rowdata_process_util, json_util
from util.metric.Gini import Gini
from util.metric.HR import HR
from util.metric.NDCG import ndcg_metric


def getHistory(each_user, dataset_name):
    history_list = []
    try:
        with open('./dataset/' + dataset_name + '/transition/' + str(each_user) + '_transition.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                history_list.append(row)
    except:
        return history_list
    return history_list


def run_sac():
    path = "./config/sacd.yaml"
    env_id = "Recommender"

    with open(path, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        name = path.split('/')[-1].rstrip('.yaml')
    cur_time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join('logs', env_id, f'{name}-seed{0}-{cur_time}')

    dataset_name = "ml-1m"
    obs_dataset = rowdata_process_util.Dataset('./dataset/' + dataset_name + '/ratings.dat')
    train_dict = obs_dataset.splitData()
    item_id_list, item_num, max_item_id = obs_dataset.getAllItem()
    item_ctr_dict = json_util.load_dict('./dataset/' + dataset_name + '/click_through_rate.json')
    pop_dict = obs_dataset.getPopular(train_dict)

    K = config['K']
    user_num = 0
    precision_test = 0
    hr_test = 0
    ndcg_test = 0
    pop_gini = copy.deepcopy(pop_dict)
    for i in pop_gini.keys():
        pop_gini[i] = 0

    for each_user in train_dict:
        user_history = getHistory(each_user, dataset_name)
        history_list_train = user_history[:int(0.8 * len(user_history))]
        history_list_test_set = user_history[int(0.8 * len(user_history)):]
        history_list_test = history_list_test_set[0:1]
        if len(history_list_train) <= 10 * 1.2:
            continue
        user_num += 1
        pass_item_list = []
        for user_id, item_id, ratings, __ in obs_dataset.data:
            if user_id == each_user and ratings == 0:
                pass_item_list.append(item_id)
        test_set = list(set([int(i[1]) for i in history_list_test_set]) - set(pass_item_list))

        observation_data = train_dict[each_user]
        env = env_sac.Env(observation_data[-K:], list(set(item_id_list)), max_item_id, each_user, K, item_ctr_dict,
                          pop_dict)
        agent = SacdAgent(env=env, log_dir=log_dir, cuda=False, state_re=True, dueling_net=False, **config)
        agent.run_offpolicy(history_list_train)
        state = history_list_test[0][0]
        actions_list = agent.exploit(state, each_user)
        precision_test += (len(set(actions_list) & set(test_set))) / (len(actions_list))
        hr_test += HR(test_set, actions_list)
        ndcg_test += ndcg_metric({each_user: actions_list}, {each_user: test_set})
        for i in actions_list:
            if i in pop_gini.keys():
                pop_gini[i] += 1

    if user_num != 0:
        print("Precision: " + str(precision_test) + " / " + str(user_num) + " = " + str(precision_test / user_num))
        print("HR: " + str(hr_test) + " / " + str(user_num) + " = " + str(hr_test / user_num))
        print("NDCG: " + str(ndcg_test) + " / " + str(user_num) + " = " + str(ndcg_test / user_num))
        for k in pop_gini.copy():
            if pop_gini[k] == 0:
                del pop_gini[k]
        print("Gini: " + str(Gini(pop_gini)))


if __name__ == '__main__':
    run_sac()
