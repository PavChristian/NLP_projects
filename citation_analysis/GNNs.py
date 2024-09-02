'''
torch must be 2.2.0 or lower to be compatible with dgl
update scipy
'''
import dgl, torch, random
import json, logging, sys, os, itertools, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import dgl.function as fn
import pandas as pd
from rclone_python import rclone
from dgl.nn import SAGEConv



class nn_files:
    def __init__(self, corpus_path, id_path, sampler_path,
                 log_file):
        #Path to wos files
        self.corpus_path = corpus_path
        #Path to a folder where ids should be saved
        self.id_path = id_path 
        #Path to sampler files 
        self.sampler_path = sampler_path 
        #Import logs 
        self.log_file = log_file
        

def obtain_ids_from_sample(self, trial_name):
    '''
    This function should be used on a subsample of data
    self (obj) -- class value 
    trial_name (str) -- name of the trial from which an article dictionary is pulled
    NOTE: Only apply to samples within the same year
    Returns 
    A dictionary with new unique ids
    DGL graph network objects only allow using numbers as ids (as opposed to strings)
    '''
    #Creating new ids
    pair_df = pd.DataFrame(columns = ['id1', 'id2', "year"])
    cor_dict = {}
    reformed_dict = {}
    #Read the author dictionary
    with open(f"{self.sampler_path}/temp/{trial_name}/temp_authors/auths_from_{year}.json", "r") as file:
        data = json.load(file)
    id_list = [r_id.split("_")[0] for r_id in list(data.keys())]
    id_set = list(set(id_list))
    for i, r_id in enumerate(id_set):
        cor_dict[r_id] = i
    for key in data.keys():
        fake_key = key.split("_")[0]
        year = key.split("_")[1]
        for co_au in data[key]['year_co_auths']:
                if co_au != "":
                    new_row = pd.DataFrame([{"id1": cor_dict[fake_key], "id2" : cor_dict[co_au],
                                             "year": year}])
                    pair_df = pd.concat([pair_df, new_row], ignore_index = True)
    #Rejoin new id numbers
    #Effectively reorganizes the old dictionary by the new id
    for key in data.keys():
        fake_key = key.split("_")[0]
        reformed_dict[cor_dict[fake_key]] = copy.deepcopy(data[key])
        reformed_dict[cor_dict[fake_key]]['r_id'] = copy.deepcopy(fake_key)
    return pair_df, reformed_dict, data

def get_new_ids(key, d):
    '''
    key (str) -- original r_id
    d (obj) -- reference dictionary 
    Converts old r_ids into numeric ids 
    '''
    return d[key]
    
def obtain_raw_wos_ids(self, year):
    '''
    Collects r_ids directly from the Web of Science corpus (rather than sub-samples)
    Useful for creating network objects from the original dataset
    January version
    Takes a long time to run
    '''
    pair_df = pd.DataFrame(columns = ['id1', 'id2', "year"])
    cor_dict = {}
    folder = f"{self.corpus_path}/{year}/authors"
    files = [os.path.join(folder, file['Path']) for file in rclone.ls(folder)]
    for file in files:
        print(f"Working on {file}", flush = True)
        with open(file, "r") as f:
            data = json.load(f)
        for key in data.keys():
            if key not in cor_dict:
                cor_dict[key] = 1
            co_authors = copy.deepcopy(data[key]['collabs'])
            for co_au in co_authors:
                if co_au != "":
                    new_row = pd.DataFrame([{"id1": key, "id2" : co_au,
                                            "year": year}])
                    pair_df = pd.concat([pair_df, new_row], ignore_index = True)
                if pair_df.shape[0] % 10000 == 0:
                    print(pair_df.shape[0], flush = True)
                
    print("Updating dataframe", flush = True)
    pair_df['id1'] = pair_df['id1'].apply(lambda x: get_new_ids(x, cor_dict))
    pair_df['id2'] = pair_df['id2'].apply(lambda x: get_new_ids(x, cor_dict))
    return pair_df, cor_dict
    
                        

def create_dgl_graph(df, src_col, dst_col, weight_col=None):
    '''
    df (obj) -- dataframe from the previous function (dyadic pairs)
    src_obj (str) -- name of the origin column 
    dst_col (str) -- name of the destination column
    weight_col (str) -- name of the weight column (if present)
    
    '''
    df[src_col] = df[src_col].astype(np.int64)
    df[dst_col] = df[dst_col].astype(np.int64)
    src = df[src_col].values
    dst = df[dst_col].values
    graph = dgl.graph((src, dst))
    
    if weight_col:
        weights = torch.tensor(df[weight_col].values, dtype=torch.float32)
        graph.edata['weight'] = weights
    
    return graph

def check_graph_sanity(graph):
    '''
    graph (str) -- a dgl graph object 
    Checks various attributes of the graph 
    '''    
    print(graph.num_edges())
    print(graph.num_nodes())
    #Out Degrees of the central node
    print(graph.out_degrees(0))
    #in Degrees of the central node 
    print(graph.in_degrees(0))
    #The latter two should be the same since is the graph is not directed
    
    

def sum_values(d, n, k):
    '''
    Sums values less than n (used in the next function)
    d (obj) -- dictionary
    n (int) -- number
    k (int) -- step back
    '''
    return sum(value for key, value in d.items() if int(key) < n and int(key) >= n - k)

def replace_nan(value):
    """
    Replaces NaN with zero.
    
    Parameters:
    value: The input value which may be a float or any type.
    
    Returns:
    The original value if it is not NaN, or zero if it is NaN.
    """
    if value != value:  # NaN is not equal to itself
        return 0
    return value

def create_node_features(auth_meta, year, rand_gender = True):
    '''
    Turns author metadata into node features (arranged in tensors)
    auth_meta (str) -- a dictionary with author metadata
    year (int) -- target year of the sample
    rand_gender (bool) -- should a random gender dummy be assigned
    Only use this on smaller samples (writing to disk is advised for larger datasets)
    '''
    features_dict = {}
    #Iterate over the dictionary
    for key, value in auth_meta.items():
        if key.split("_")[1] != str(year):
            continue
        #Turn citations into a tensor  
        citations = value.get('yearly_cit')
        sorted_values = [citations[key] for key in sorted(citations.keys())]
        #Convert the list of sorted values to a tensor
        feat_tensor = torch.tensor(sorted_values)
        #Previous 3 years (non_inclusive)
        t_minus_3 = torch.tensor([replace_nan(sum_values(citations, year, 3))])
        feat_tensor = torch.cat((feat_tensor, t_minus_3))
        #Previous 5 years
        t_minus_5 = torch.tensor([replace_nan(sum_values(citations, year, 5))])
        feat_tensor = torch.cat((feat_tensor, t_minus_5))
        #Total citations (over all time)
        total_citations = torch.tensor([sum(value for key, value in citations.items())])
        feat_tensor = torch.cat((feat_tensor, total_citations))
        #Add a random gender dummy
        if rand_gender: 
            gender = copy.deepcopy(random.random())
            auth_meta[key]['gender'] = gender
            feat_tensor = torch.cat((feat_tensor, torch.tensor([gender])))
        #Academic age
        ac_age = int(auth_meta[key]['year']) - int(auth_meta[key]['fy'])
        feat_tensor = torch.cat((feat_tensor, ac_age))
        features_dict[key] = feat_tensor
        
    return auth_meta, features_dict
    
def check_addresses_old(ad1, ad2):
    '''
    Checks whether two authors have the same zipcode, insitution
    ad1 (dict) -- addresses of the first author in a dyad 
    ad2 (dict) -- addresses of the second author in a dyad
    NOTE: This works only on the uncleaned address dictionary (addresses field in auth_dict)
    '''
    z_dummy = 0
    i_dummy = 0
    for a1 in ad1:
        zip1 = a1.get('zip')
        inst1 = a1.get('organization').split(";")[1]
        for a2 in ad2:
            zip2 = a2.get('zip')
            inst2 = a2.get('organization').split(";")[1]
            if zip1 == zip2 and zip1 != "" and z_dummy == 0:
                z_dummy = 1
            else:
                pass
            if inst1 == inst2 and inst1 != "" and i_dummy == 0:
                i_dummy = 1
            else:
                pass
    return [z_dummy, i_dummy]

    
def create_edge_features(auth_meta, pair_df, feature_data, author_similarity, write_path):
    '''
    Using author metadata, create edge features for training
    auth_meta (dict) -- output from create_node_features
    pair_df (df) -- output from obtain ids from sample
    feature_data (dict) -- dictionary with the necessary features (it should come from the co-citation file)
    Immediately writing to disk to avoid memory problems
    '''
    i = 0
    with open(write_path) as json_file:
        for index, row in pair_df.iterrows():
            id1, id2, year = row['id1'], row['id2'], row['year']
            ad_values = check_addresses_old(auth_meta[id1],
                                            auth_meta[id2])
            #Dummies for similar addresses
            feat_tensor = torch.tensor(ad_values)
            #finding unique id
            unique_id = id1 + "+" + id2 + "_" + year
            selected_keys = ['com_cit', "co-cit", "fr_cit"]
            mini_dict = [feature_data[key] for key in feature_data[unique_id] if key in selected_keys]
            feat_tensor = torch.cat(feat_tensor, torch.tensor(mini_dict))
            feat_tensor = torch.cat(feat_tensor, torch.tensor([author_similarity[unique_id]]))
            #Appending the dictionary to a list
            json_line = json.dumps({id1: {"pair": id2, "edge_features": feat_tensor}})
            json_file.write(json_line + "/n")  
            i +=1 
            if i % 10000 == 0:
                print(f"{i} iterations have been completed")
    
def assign_random_features(graph, num_node_feat, num_edge_feat):
    '''
    NOTE: This function is only for testing and small samples
    Assigns random features to the graph (should later be replaced with actual features)
    graph (obj) -- a dgl graph object
    num_node_feat (int) -- number of node features
    num_edge_feat (int) -- number of edge features
    Matrices can also be assigned 
    '''
    n_nodes = graph.num_nodes()
    n_edges = graph.num_edges()
    graph.ndata['feat'] = torch.randn(n_nodes, num_node_feat)
    graph.edata['feat'] = torch.randn(n_edges, num_edge_feat)
    return graph

def pre_proc_and_split(g, test_prop = 0.1):
    '''
    NOTE: This function is only for testing and small samples. Do not use on large samples/graphs
    Create test and train splits for GNN training
    g (obj) -- a dgl graph object 
    test_prop (float) -- proportion of observations in the test sample
    '''
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * test_prop)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    #Remove test set egdes
    train_g = dgl.remove_edges(g, eids[:test_size])
    #Create positive and negative edges for training 
    #Positive edges -- edges where a link between two nodes exists 
    #Negative edges -- two nodes without an edge
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        
        
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


def launch_training_loop(model, pred, optimizer, train_g, train_pos_g, train_neg_g,
                         test_pos_g, test_neg_g):
    all_logits = []
    for e in range(100):
        # forward
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {}'.format(e, loss))
    #Check results
    from sklearn.metrics import roc_auc_score
    with torch.no_grad():
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('AUC', compute_auc(pos_score, neg_score))


def main():
    pavel_network = nn_files(corpus_path= "modal_code/wos/file_system/jsons",
                             id_path = None,
                             sampler_path = "modal_code/sampler",
                             log_file = "modal_code/graph_nets/logs/log_out.log")
    logging.basicConfig(filename=pavel_network.log_file, level=logging.INFO)
    sys.stdout = sys.stderr = open(pavel_network.log_file, 'a')
    with open(pavel_network.log_file, 'w'):
        pass       
    pair_df, ref_dict, auth_dict = obtain_ids_from_sample(pavel_network, "trial2")
    g = create_dgl_graph(pair_df, "id1", "id2")
    check_graph_sanity(g)
    auth_dict, features_dict = create_node_features(auth_dict, 2010)
    print(features_dict)
    g = assign_random_features(g, 3, 3)
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = pre_proc_and_split(g)
    #Create the model
    model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
    #Define calculation method 
    pred = DotPredictor()
    #Define the optimizer
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    #Run the training process
    launch_training_loop(model, pred, optimizer, train_g, train_pos_g, train_neg_g,
                         test_pos_g, test_neg_g)
    
    
    
    
if __name__ == "__main__":
    main()
    print("Done")
    
    
        
        