"""utility module which provides dictionary manipulation"""

def get_elem(d, lst):
    """get dict element by accessing its tags in the order given by lst
    Input:
        d   : (dict) dictionary accessed
        list: (list) order to access dict tags
    Output:
        data: element of d"""

    try:
        for tag in lst:
            d = d[tag]
    except:
        d = None
    return d