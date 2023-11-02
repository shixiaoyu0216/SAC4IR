class Dataset():
    def __init__(self, data_path, split="::"):
        self.data = self.loadData(data_path, split="::")
        self.train_dict = {}
        self.test_dict = {}
        self.item_id_list = []

    def loadData(self, dataset_path, split="::"):
        data_item_list = []
        for data_item in open(dataset_path):
            temp_tuple = list(data_item.strip().split(split)[:4])
            temp_tuple[0] = int(temp_tuple[0])
            temp_tuple[1] = int(temp_tuple[1])
            temp_tuple[2] = int(temp_tuple[2])
            temp_tuple[3] = int(temp_tuple[3])
            data_item_list.append(tuple(temp_tuple))
        data_item_list = sorted(data_item_list, key=lambda tup: tup[3])
        data_item_list = sorted(data_item_list, key=lambda tup: tup[0])
        return data_item_list

    def splitData(self):
        userid_dict, train_dict = {}, {}
        for user_id, item_id, _, __ in self.data:
            if user_id not in userid_dict:
                userid_dict[user_id] = list()
            userid_dict[user_id].append(item_id)

        for user_id in userid_dict:
            train_dict[user_id] = userid_dict[user_id][:]
        self.train_dict = train_dict
        return train_dict

    def getAllItem(self):
        item_list = []
        for _, item_id, __, ___ in self.data:
            item_list.append(item_id)
        item_list = list(set(item_list))
        item_list.sort()
        item_num = len(item_list)
        max_item = max(item_list)
        self.item_id_list = item_list
        return item_list, item_num, max_item

    def getPopular(self, data_dict_train):
        temp_dict = {}
        for user, items in data_dict_train.items():
            temp_dict[user] = data_dict_train[user]
        item_popularity = dict()
        for user, items in temp_dict.items():
            for item in items:
                if item not in item_popularity:
                    item_popularity[item] = 0
                item_popularity[item] += 1
        return item_popularity
