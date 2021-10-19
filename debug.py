import pandas as pd
import dgl
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        #Here we go from our original number of features to the size of our hidden representations
        self.conv1 = GraphConv(in_feats, h_feats)
        #Here we go from the hidden representation to the dimension of the number of classes for the probabilities
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat) #Here we apply the first convolution
        h = F.relu(h) #Here we apply the selected non-linearity. In this case a Relu
        h = self.conv2(g, h) #Here we apply the second convolution
        return h

def get_results(y_pred_probas, y_test, threshold=0.5):
    pred_probas_1 = y_pred_probas[:,1]
    preds_1 = np.where(pred_probas_1>threshold,1,0)
    auc = roc_auc_score(y_test, pred_probas_1)
    f1 = f1_score(y_test,preds_1)
    prec = precision_score(y_test,preds_1)
    recall = recall_score(y_test,preds_1)
    return auc, f1, prec, recall

def create_masks(df, seed=23, test_size=0.2):
    '''
    This function creates binary tensors that indicate whether an user is on the train or test set
    '''
    np.random.RandomState(seed)
    temp_df = df.copy()
    temp_df['split_flag'] = np.random.random(df.shape[0])
    train_mask = th.BoolTensor(np.where(temp_df['split_flag'] <= (1 - test_size), True, False))
    test_mask = th.BoolTensor(np.where((1 - test_size) < temp_df['split_flag'] , True, False))
    return train_mask, test_mask


df = pd.read_csv('data/nodes_features_workshop.csv').set_index('USER_ID')

#Create binary masks
train_mask, test_mask = create_masks(df, 23, 0.3)

print(train_mask)

#Here we transform the tensors so they indicate the indices of the train and test users instead of the binary
train_nid = train_mask.nonzero().squeeze()
test_nid = test_mask.nonzero().squeeze()

print(train_nid)

#Create X and Y dataframes
X = df.drop(['FRAUD'], axis=1)
y = df.drop(['DEVICE_TYPE','EXPECTED_VALUE','SALES'], axis=1)

print('The shape of the X DataFrame is: ',X.shape)
print('The shape of the y DataFrame is: ',y.shape)

#Transform the X and Y dataframes to tensors now as well
X = th.tensor(X.values).float()
y = th.tensor(y.values).type(th.LongTensor).squeeze_()

print(X.shape)
print(y.shape)

hidden_size = 16
num_classes = 2

# Create the model with given dimensions
model = GCN(X.shape[1], hidden_size, num_classes)

def train(g, features, labels, train_mask, test_mask, epochs, model):
    optimizer = th.optim.Adam(model.parameters(), lr=0.01) #Selected optimizer
    best_val_acc = 0
    best_test_acc = 0

    #Here we create the validation set with a portion of the train set
    val_mask = train_mask[:len(train_mask) // 5]
    train_mask = train_mask[len(train_mask) // 5:]

    train_mask = train_mask.nonzero().squeeze()
    test_mask = test_mask.nonzero().squeeze()
    val_mask = val_mask.nonzero().squeeze()

    for e in range(epochs):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
    
    #Check results on test set
    y_pred_probas = model(g, features).detach().numpy()
    y_pred_probas = y_pred_probas[test_mask]
    y_preds = y_pred_probas[:,1]
    model_name = 'GCN'
    
    auc, f1, prec, recall = get_results(y_pred_probas, labels[test_nid], 0.5)
    dict_results = {'Model':model_name, 'AUC':auc, 'F1 Score':f1, 'Precision':prec, 'Recall':recall}
    results_df = pd.DataFrame(columns=['Model','AUC','F1 Score','Precision','Recall'])
    results_df = results_df.append(dict_results, ignore_index=True)
    return results_df

#We start by using the same edges df but transforming the from and to columns into arrays
edges_df = pd.read_csv('data/new_edges_workshop.csv')

src = edges_df['~from'].to_numpy()
snk = edges_df['~to'].to_numpy()

G_dgl = dgl.graph((src,snk))
G_dgl.add_edges(G_dgl.nodes(), G_dgl.nodes())

epochs = 100
results_df = train(G_dgl, X, y, train_mask, test_mask, epochs, model)

print('This is a test')