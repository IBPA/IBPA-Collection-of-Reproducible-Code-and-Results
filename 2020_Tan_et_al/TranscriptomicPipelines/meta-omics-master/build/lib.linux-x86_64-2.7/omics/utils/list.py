"""utility module which provides list manipulation"""

def split_list(lst, length):
    """return a list of lists with the specified length"""
    batch_list  = []
    batch       = []
    for elem in lst:
        batch.append(elem)
        if len(batch) == length:
            batch_list.append(batch)
            batch = []
    if batch:
        batch_list.append(batch)
    return batch_list

def intersect(lst_1, lst_2):
    """return the union """
    return list(set(lst_1) & set(lst_2))

def concatenate(lst):
    return ';'.join([str(item) for item in lst])
