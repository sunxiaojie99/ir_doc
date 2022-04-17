import imp


import os


def divide_passage_collection(dir_name, new_dir_name, old_num, new_num):
    """重新划分passgae集合的文件数量
    dir_name: dureader-retrieval-baseline-dataset/passage-collection
    """
    list_dir = os.listdir(dir_name)
    list_dir = list(dir_name+'/'+i for i in list_dir) 
    print(list_dir)



divide_passage_collection('src')