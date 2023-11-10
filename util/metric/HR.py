def HR(test_set, rec_list):
    if list(set(test_set) & set(rec_list)):
        return 1
    else:
        return 0
