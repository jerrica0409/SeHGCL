import pickle as pkl
from scipy.sparse import rand
import dgl
import torch


def load_movielens():

    u_i_src = []
    u_i_dst = []
    with open("data/movielens/train.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(" ")
            user, item, rate = int(_line[0]), int(_line[1]), int(_line[2])
            u_i_src.append(user)
            u_i_dst.append(item)

    i_g_src = []
    i_g_dst = []
    with open("data/movielens/movie_genre.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(" ")
            item, gen = int(_line[0]), int(_line[1])
            i_g_src.append(item)
            i_g_dst.append(gen)
        # i_g_src = [x - 1 for x in i_g_src]
        # i_g_dst = [x - 1 for x in i_g_dst]


    hg = dgl.heterograph({
        ('item', 'iu', 'user'): (u_i_dst, u_i_src),
        ('user', 'ui', 'item'): (u_i_src, u_i_dst),
        ('item', 'ig', 'gen'): (i_g_src, i_g_dst),
        ('gen', 'gi', 'item'): (i_g_dst, i_g_src)
    }
    )

    edges_dict = {
        ('item', 'iu', 'user'): (u_i_dst, u_i_src),
        ('user', 'ui', 'item'): (u_i_src, u_i_dst),
        ('item', 'ig', 'gen'): (i_g_src, i_g_dst),
        ('gen', 'gi', 'item'): (i_g_dst, i_g_src)
    }

    nodes_dict = {
        "user": hg.num_nodes("user"),
        "item": hg.num_nodes("item"),
        "gen": hg.num_nodes("gen")
    }

    hg_processed = dgl.heterograph(
        data_dict=edges_dict, num_nodes_dict=nodes_dict
    )

    meta_paths = {
        'user': [['ui', 'iu'],['ui','ig','gi','iu']],#,['ua','au'],['uo','ou']
        'item': [['iu', 'ui'],['ig','gi']]#
    }

    user_key = 'user'
    item_key = 'item'
    user_num = hg.num_nodes("user")
    item_num = hg.num_nodes("item")

    return hg_processed, meta_paths, user_key, item_key, user_num, item_num


def load_amazon():

    u_i_src = []
    u_i_dst = []
    with open("data/amazon/user_item.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split("\t")
            user, item, rate = int(_line[0]), int(_line[1]), int(_line[2])
            u_i_src.append(user)
            u_i_dst.append(item)

    i_b_src = []
    i_b_dst = []
    with open("data/amazon/item_brand.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(",")
            item, brand = int(_line[0]), int(_line[1])
            i_b_src.append(item)
            i_b_dst.append(brand)

    i_c_src = []
    i_c_dst = []
    with open("data/amazon/item_category.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(",")
            item, category = int(_line[0]), int(_line[1])
            i_c_src.append(item)
            i_c_dst.append(category)


    hg = dgl.heterograph({
        ('item', 'iu', 'user'): (u_i_dst, u_i_src),
        ('user', 'ui', 'item'): (u_i_src, u_i_dst),
        # ('item', 'ib', 'brand'): (i_b_src, i_b_dst),
        # ('brand', 'bi', 'item'): (i_b_dst, i_b_src),
        ('item', 'ic', 'category'): (i_c_src, i_c_dst),
        ('category', 'ci', 'item'): (i_c_dst, i_c_src)
    }
    )

    edges_dict = {
        ('item', 'iu', 'user'): (u_i_dst, u_i_src),
        ('user', 'ui', 'item'): (u_i_src, u_i_dst),
        # ('item', 'ib', 'brand'): (i_b_src, i_b_dst),
        # ('brand', 'bi', 'item'): (i_b_dst, i_b_src),
        ('item', 'ic', 'category'): (i_c_src, i_c_dst),
        ('category', 'ci', 'item'): (i_c_dst, i_c_src)
    }

    nodes_dict = {
        "user": hg.num_nodes("user"),
        "item": hg.num_nodes("item"),
        # "brand": hg.num_nodes("brand"),
        "category": hg.num_nodes("category")
    }

    hg_processed = dgl.heterograph(
        data_dict=edges_dict, num_nodes_dict=nodes_dict
    )

    meta_paths = {
        'user': [['ui', 'iu'],['ui','ic','ci','iu']],
        'item': [['iu', 'ui'],['ic','ci']]#,['ib','bi']
    }

    user_key = 'user'
    item_key = 'item'
    user_num = hg.num_nodes("user")
    item_num = hg.num_nodes("item")

    return hg_processed, meta_paths, user_key, item_key, user_num, item_num


def load_yelp():

    u_b_src = []
    u_b_dst = []
    with open("data/yelp/user_business.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(" ")
            user, business, rate = int(_line[0]), int(_line[1]), int(_line[2])
            u_b_src.append(user)
            u_b_dst.append(business)


# business_category   a:category
    b_a_src = []
    b_a_dst = []
    with open("data/yelp/business_category.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(" ")
            business, category = int(_line[0]), int(_line[1])
            b_a_src.append(business)
            b_a_dst.append(category)

# business_city   i:city
    b_i_src = []
    b_i_dst = []
    with open("data/yelp/business_city.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(" ")
            business, city = int(_line[0]), int(_line[1])
            b_i_src.append(business)
            b_i_dst.append(city)


    hg = dgl.heterograph({
        ('user', 'ub', 'business'): (u_b_src, u_b_dst),
        ('business', 'bu', 'user'): (u_b_dst, u_b_src),
        ('business', 'ba', 'category'): (b_a_src, b_a_dst),
        ('category', 'ab', 'business'): (b_a_dst, b_a_src),
        ('business', 'bi', 'city'): (b_i_src, b_i_dst),
        ('city', 'ib', 'business'): (b_i_dst, b_i_src)
    }
    )

    edges_dict = {
        ('user', 'ub', 'business'): (u_b_src, u_b_dst),
        ('business', 'bu', 'user'): (u_b_dst, u_b_src),
        ('business', 'ba', 'category'): (b_a_src, b_a_dst),
        ('category', 'ab', 'business'): (b_a_dst, b_a_src),
        ('business', 'bi', 'city'): (b_i_src, b_i_dst),
        ('city', 'ib', 'business'): (b_i_dst, b_i_src)
    }

    nodes_dict = {
        "user": hg.num_nodes("user"),
        "business": hg.num_nodes("business"),
        "category": hg.num_nodes("category"),
        "city": hg.num_nodes("city")
    }

    hg_processed = dgl.heterograph(
        data_dict=edges_dict, num_nodes_dict=nodes_dict
    )

    meta_paths = {
        'user': [['ub', 'bu']],#,['uo','ou']
        'business': [['bu', 'ub'],['ba','ab'],['bi','ib']]
    }

    user_key = 'user'
    item_key = 'business'
    user_num = hg.num_nodes("user")
    item_num = hg.num_nodes("business")

    return hg_processed, meta_paths, user_key, item_key, user_num, item_num


def load_lastfm():

    u_a_src = []
    u_a_dst = []
    with open("data/lastfm/train.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(" ")
            user, artist, rate = int(_line[0]), int(_line[1]), int(_line[2])
            u_a_src.append(user)
            u_a_dst.append(artist)

    a_t_src = []
    a_t_dst = []
    with open("data/lastfm/artist_tag.dat") as fin:
        for line in fin.readlines():
            _line = line.strip().split(" ")
            artist, tag = int(_line[0]), int(_line[1])
            a_t_src.append(artist)
            a_t_dst.append(tag)

    hg = dgl.heterograph({
        ('user', 'ua', 'artist'): (u_a_src, u_a_dst),
        ('artist', 'au', 'user'): (u_a_dst, u_a_src),
        ('artist', 'at', 'tag'): (a_t_src, a_t_dst),
        ('tag', 'ta', 'artist'): (a_t_dst, a_t_src)
    }
    )

    edges_dict = {
        ('user', 'ua', 'artist'): (u_a_src, u_a_dst),
        ('artist', 'au', 'user'): (u_a_dst, u_a_src),
        ('artist', 'at', 'tag'): (a_t_src, a_t_dst),
        ('tag', 'ta', 'artist'): (a_t_dst, a_t_src)
    }

    nodes_dict = {
        "user": hg.num_nodes("user"),
        "artist": 17632,  #hg.num_nodes("artist"),
        "tag": hg.num_nodes("tag")
    }

    hg_processed = dgl.heterograph(
        data_dict=edges_dict, num_nodes_dict=nodes_dict
    )

    meta_paths = {
        'user': [['ua', 'au']],
        'artist': [['au', 'ua'],['at','ta']]
    }

    user_key = 'user'
    item_key = 'artist'
    user_num = hg.num_nodes("user")
    item_num = hg.num_nodes("artist")

    return hg_processed, meta_paths, user_key, item_key, user_num, item_num


def load_g(dataset):
    if dataset == "movielens":
        g, metapaths, user_key, item_key, user_num, item_num = load_movielens()
    elif dataset == "amazon":
        g, metapaths, user_key, item_key, user_num, item_num = load_amazon()
    elif dataset == "yelp":
        g, metapaths, user_key, item_key, user_num, item_num = load_yelp()
    elif dataset == "lastfm":
        g, metapaths, user_key, item_key, user_num, item_num = load_lastfm()
    return g, metapaths, user_key, item_key, user_num, item_num

# if __name__=="__main__":
#     load_g()