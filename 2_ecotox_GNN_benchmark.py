
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import pandas as pd
import numpy as np
from torch.utils import data
from torch_geometric.data import Data
from models import GCN_MLP_combined_mod
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report
from collections import defaultdict
import os
import wandb
from args_parser import arg_parser_ecotox
from torch.optim.lr_scheduler import StepLR
from eco_dataloader import Data_eco_variable




def load_interaction(filename):
    edges = np.loadtxt(filename, delimiter=',')
    split_edges = np.vstack([edges[:,0].flatten(), edges[:,1].flatten(), edges[:,2].flatten(), edges[:,4].flatten()])
    pos_edges = edges[edges[:,4]==1]
    neg_edges = edges[edges[:,4]==0]
    pos_arr = np.vstack([pos_edges[:,0].flatten(), pos_edges[:,1].flatten(), pos_edges[:,2].flatten()])
    neg_arr = np.vstack([neg_edges[:,0].flatten(), neg_edges[:,1].flatten(), neg_edges[:,2].flatten()])
    num_pos = pos_arr.shape[1]
    num_neg = neg_arr.shape[1]
    print(num_pos, num_neg)
    return pos_arr, neg_arr, split_edges

def load_fmat_u_v(xu_filename, xv_filename):
    xu_mat = np.load(xu_filename)
    
    xv_mat = np.load(xv_filename)

    return xu_mat, xv_mat

def load_graph_with_uv_efeat(xu_mat, xv_mat, links1, links2):
    # print(links1.shape, links2.shape)
    edge_index1_1way = torch.Tensor([links1[0,:], links1[1,:]]).to(torch.int64)
    edge_index2_1way = torch.Tensor([links2[0,:], links2[1,:]]).to(torch.int64)
    edge_index_1way = torch.cat([edge_index1_1way, edge_index2_1way], dim=1)

    edge_feats1_1way = torch.Tensor(links1[2,:])
    edge_feats2_1way = torch.Tensor(links2[2,:])
    edge_feats_1way = torch.cat((edge_feats1_1way, edge_feats2_1way), dim=0)

    edge_index1_2way  = (two_way_edges(torch.Tensor(links1[0,:]).to(torch.int64), torch.Tensor(links1[1,:]).to(torch.int64))).to(torch.int64)
    edge_index2_2way  = (two_way_edges(torch.Tensor(links2[0,:]).to(torch.int64), torch.Tensor(links2[1,:]).to(torch.int64))).to(torch.int64)    
    edge_index_2way  = torch.cat([edge_index1_2way, edge_index2_2way], dim=1)

    edge_feats1_2way = torch.Tensor(np.concatenate((links1[2,:], links1[2,:])))
    edge_feats2_2way = torch.Tensor(np.concatenate((links2[2,:], links2[2,:])))
    edge_feats_2way = torch.cat((edge_feats1_2way, edge_feats2_2way), dim=0)
    
    xu_mat = torch.Tensor(xu_mat)
    xv_mat =  torch.Tensor(xv_mat)

    data_graph = Data(xu = xu_mat, xv = xv_mat, edge_index = edge_index_1way, edge_index1 = edge_index_2way, efeats = edge_feats_1way, efeats1 = edge_feats_2way)   
    return data_graph

def two_way_edges(u,v):
    return torch.stack([torch.cat([u,v]), torch.cat([v,u])], 0)


def train_gcnmlp_efts(train_graph, train_loader, model, criterion, optimizer, device):
    model.train()
    losses = []
    aucs = []
    accuracies = []
    pbar = train_loader
    for i,data in enumerate(pbar):
        optimizer.zero_grad()
        # print(train_graph.xu.shape, train_graph.xv.shape, train_graph.edge_index1.shape, train_graph.efeats1.shape)
        # z = model.encode(train_graph.xu, train_graph.xv, train_graph.edge_index1, train_graph.efeats1) 
        all_edge_inds = torch.cat([data[0][:,:2].to(int), data[1][:,:2].to(int)]).T.to(device)
        all_d = torch.cat((data[0][:,2], data[1][:,2]), dim=0).to(torch.float32).to(device)
        z = model.encode(train_graph.xu, train_graph.xv, all_edge_inds, train_graph.edge_index1, all_d.unsqueeze(1))#train_graph.efeats1)   
        # all_edge_inds = torch.cat([data[0][:,:2].to(int), data[1][:,:2].to(int)])
        num_edges = data[0].shape[0]
        all_edge_labels = torch.cat([torch.Tensor([1]*num_edges), torch.Tensor([0]*num_edges)]).to(device)
        # out = model.decode(z, all_edge_inds).sigmoid().view(-1)
        out = z.squeeze()
        loss = criterion(out, all_edge_labels)
        auc = roc_auc_score(all_edge_labels.cpu().detach().numpy(), out.cpu().detach().numpy())
        losses.append(loss.item())
        aucs.append(auc.item())    
        loss.backward()
        optimizer.step()
        # threshold the predictions at 0.5 and compare with the true labels
        preds = (out > 0.5).long()
        acc = (preds == all_edge_labels).float().mean()
        accuracies.append(acc.item())
    return np.mean(losses), np.mean(aucs), np.mean(accuracies)


def val_gcnmlp_efts(data_graph, data_loader, model, criterion, device):
    model.eval()
    losses = []
    aucs = []
    accuracies = []
    pbar = data_loader
    for i,data in enumerate(pbar):

        all_edge_inds = torch.cat([data[0][:,:2].to(int), data[1][:,:2].to(int)])
        val_edges_1way = torch.stack([all_edge_inds[:,0], all_edge_inds[:,1]], axis=1).T.to(device)
        val_edges = torch.stack([torch.cat([all_edge_inds[:,0], all_edge_inds[:,1]]), torch.cat([all_edge_inds[:,1], all_edge_inds[:,0]])], 0).to(device)
        net_graph_edge_inds = torch.cat(([data_graph.edge_index1, val_edges]),dim=1).to(device)
        val_efeats = torch.cat((data[0][:,2], data[1][:,2], data[0][:,2], data[1][:,2]), dim=0).to(torch.float32).to(device)
        val_efeats_1way = torch.cat((data[0][:,2], data[1][:,2]), dim=0).to(torch.float32).to(device)
        net_graph_efeats = torch.cat((data_graph.efeats1, val_efeats))
        # z = model.encode(data_graph.xu, data_graph.xv, net_graph_edge_inds, net_graph_efeats)#, train_graph.efeats1) 

        z = model.encode(data_graph.xu, data_graph.xv, val_edges_1way, net_graph_edge_inds, val_efeats_1way.unsqueeze(1))#net_graph_efeats)  
        num_edges1, num_edges2 = data[0].shape[0], data[1].shape[0]
        all_edge_labels = torch.cat([torch.Tensor([1]*num_edges1), torch.Tensor([0]*num_edges2)]).to(device)
        # out = model.decode(z, all_edge_inds).sigmoid().view(-1)
        out = z.squeeze()
        loss = criterion(out, all_edge_labels)
        auc = roc_auc_score(all_edge_labels.cpu().detach().numpy(), out.cpu().detach().numpy())
        losses.append(loss.item())
        aucs.append(auc.item())    
        # threshold the predictions at 0.5 and compare with the true labels
        preds = (out > 0.5).long()
        acc = (preds == all_edge_labels).float().mean()
        accuracies.append(acc.item())
    return np.mean(losses), np.mean(aucs), np.mean(accuracies)

def predict_from_model(graph, data_loader, model, criterion, device):
    model.eval()
    losses = []
    
    given_labels = []
    given_efeats = []
    given_species = []
    given_chemicals = []
    predicted_scores = []
    # predicted_labels = []
    
    for _, _, links in data_loader:
        batch_links = links.T
        target = (batch_links[3,:]).to(device)
        val_edges_1way = batch_links[:2,:].to(torch.int64).to(device)
        net_graph_edge_inds = torch.cat(([graph.edge_index1, batch_links[:2,:].to(torch.int64).to(device)]),dim=1)
        net_graph_efeats = torch.cat((graph.efeats1, batch_links[2,:].to(torch.float32).to(device)))
        val_efeats_1way = batch_links[2,:].to(torch.float32).to(device)
        z = model.encode(graph.xu, graph.xv, val_edges_1way, net_graph_edge_inds, val_efeats_1way.unsqueeze(1))
        # z = model.encode(graph.xu, graph.xv, net_graph_edge_inds, net_graph_efeats)
        # outputs = model.decode(z, batch_links[:2,:].to(torch.int64).to(device).T).sigmoid().view(-1)
        outputs = z.squeeze()
        loss = criterion(outputs, target)
        losses.append(loss.item())
        

        given_labels.extend(target.cpu().detach().numpy().tolist())
        given_species.extend(links[:,0].to(int).tolist())
        given_chemicals.extend(links[:,1].to(int).tolist())
        given_efeats.extend(links[:,2].tolist())
        predicted_scores.extend(outputs.cpu().detach().numpy().flatten())
        # predicted_labels.extend(preds.cpu().detach().numpy().tolist())
    # Convert predictions to binary (0 or 1) based on threshold 0.5
    binary_preds = (np.array(predicted_scores) >= 0.5).astype(int)

    # Compute accuracy and ROC AUC score
    acc = accuracy_score(np.array(given_labels), binary_preds)
    auc = roc_auc_score(np.array(given_labels), np.array(predicted_scores))
    given_labels_arr = np.array(given_labels).astype(int)
    # print(given_labels_arr)
    print(classification_report(given_labels_arr, binary_preds))
    return np.mean(losses), auc, acc, given_labels, predicted_scores, binary_preds, given_species,given_chemicals, given_efeats


def train_prediction(graph, data_loader, model, criterion, optimizer, device):
    model.train()
    all_preds = [] 
    all_labels = []
    t_loss = 0

    for _, _, links in data_loader:
        optimizer.zero_grad()
        batch_links = links.T
        target = (batch_links[3,:]).to(device)
        net_graph_edge_inds = torch.cat(([graph.edge_index1, batch_links[:2,:].to(torch.int64).to(device)]),dim=1)
        net_graph_efeats = torch.cat((graph.efeats1, batch_links[2,:].to(torch.float32).to(device)))
        z = model.encode(graph.xu, graph.xv, net_graph_edge_inds, net_graph_efeats)
        outputs = model.decode(z, batch_links[:2,:].to(torch.int64).to(device).T).sigmoid().view(-1)
        loss = criterion(outputs, target)
        t_loss += loss.item()
        all_preds.extend(outputs.cpu().detach().numpy().flatten())
        all_labels.extend(target.cpu().detach().numpy().flatten())

        loss.backward()
        optimizer.step()

    # Convert predictions to binary (0 or 1) based on threshold 0.5
    binary_preds = (np.array(all_preds) >= 0.5).astype(int)

    # Compute accuracy and ROC AUC score
    acc = accuracy_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_preds)
    return np.mean(t_loss), acc, auc

def test_prediction(graph, data_loader, model, criterion, device):
    model.eval()
    all_preds = [] 
    all_labels = []
    test_loss = 0

    with torch.no_grad():
        for _, _, links in data_loader:
            batch_links = links.T
            target = (batch_links[3,:]).to(device)
            net_graph_edge_inds = torch.cat(([graph.edge_index1, batch_links[:2,:].to(torch.int64).to(device)]),dim=1)
            net_graph_efeats = torch.cat((graph.efeats1, batch_links[2,:].to(torch.float32).to(device)))
            z = model.encode(graph.xu, graph.xv, net_graph_edge_inds, net_graph_efeats)
            outputs = model.decode(z, batch_links[:2,:].to(torch.int64).to(device).T).sigmoid().view(-1)
            t_loss = criterion(outputs, target)
            test_loss += t_loss.item()
            # preds = (outputs >= 0.5).long()
            all_preds.extend(outputs.cpu().detach().numpy().flatten())
            all_labels.extend(target.cpu().detach().numpy().flatten())
        # Convert predictions to binary (0 or 1) based on threshold 0.5
        binary_preds = (np.array(all_preds) >= 0.5).astype(int)

        # Compute accuracy and ROC AUC score
        acc = accuracy_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds)

    return np.mean(test_loss), acc, auc



if __name__ == '__main__':
    
    args = arg_parser_ecotox()
    print(args)
    
    train_pos_edges, train_neg_edges, train_edges = load_interaction(args.train_file) 
    val_pos_edges, val_neg_edges, val_edges = load_interaction(args.val_file)
    test_pos_edges, test_neg_edges, test_edges = load_interaction(args.test_file)

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print(device)
    # device = torch.device("cpu")
    xu, xv = load_fmat_u_v(args.u_filename, args.v_filename)
    dim_u = xu.shape[1]
    dim_v = xv.shape[1]
    print(dim_u, dim_v)
    dim_d = 1

    train_graph = load_graph_with_uv_efeat(xu, xv, train_pos_edges, train_neg_edges)
    
    train_data = train_graph.to(device)
    # model = LogisticRegression(dim_u,dim_v, args.p_gcn)
    model = GCN_MLP_combined_mod(dim_u, dim_v, args.hdim1, args.hdim2, args.hdim3, args.outdim, args.p_gcn)
    model.to(device)
    # Define a learning rate scheduler
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-8)
    # scheduler = StepLR(optimizer, step_size= 20, gamma=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    total_params = sum(p.numel() for param in model.parameters() for p in param)
    # print(f'Total number of parameters is {total_params}')
    # print(summary(model))

    train_set = Data_eco_variable(train_pos_edges, train_neg_edges, train_edges)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_edges_all = [torch.from_numpy(val_pos_edges.T).to(device),torch.from_numpy(val_neg_edges.T).to(device)]
    val_set = Data_eco_variable(val_pos_edges, val_neg_edges, val_edges)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    test_set = Data_eco_variable(test_pos_edges, test_neg_edges, test_edges)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model.train()
    
    metrics = defaultdict()
    metric_per_iter = defaultdict()
    best_val_auc = final_test_auc = best_auc = 0
    ep = 0
    patience = 10
    if args.train_flag==1:
        # wandb_dir = args.wandbdir
        # wandb.init(project='gcnmlp_ecotox', dir=wandb_dir)
        # runName = wandb.run.name
        name_str = args.name_str
        # print(name_str,str(args.epochs),str(args.hdim1),str(args.hdim2),str(args.hdim3),str(args.outdim),str(args.lr),str(args.p_gcn))
        exp_name = name_str+'eco_Sagemlp_with_d_leaky'+'_ep'+str(args.epochs)+'_h1'+str(args.hdim1)+'_h2'+str(args.hdim2)+'_h3'+str(args.hdim3)+'_o'+str(args.outdim)+'_lr'+ str(args.lr)+'_pg'+ str(args.p_gcn)
        print(exp_name)
        # wandb.run.name = exp_name + "-" + runName.split("-")[-1]    
        for epoch in range(1,args.epochs+1):
            train_loss,train_auc, train_acc = train_gcnmlp_efts(train_graph, train_loader, model, criterion, optimizer, device)
            val_loss,val_auc, val_acc = val_gcnmlp_efts(train_graph, val_loader, model, criterion, device)
            test_loss,test_auc, test_acc = val_gcnmlp_efts(train_graph, test_loader, model, criterion, device)
            
            print('Epoch {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f} '.format(epoch, train_loss, train_auc, val_loss, val_auc, test_loss, test_auc))

            data  = [epoch,train_loss,train_auc,train_acc,val_loss,val_auc,val_acc,test_loss,test_auc,test_acc, total_params]
            dataName  = ["epoch","train_loss", "train_auc","train_acc", "val_loss", "val_auc","val_acc", "test_loss", "test_auc", "test_acc", "num_param"]
            metric_per_iter = {
                    'epoch': epoch,
                    'dropout': args.p_gcn,
                    'train_loss': train_loss,
                    'train_auc': train_auc,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'test_loss': test_loss,
                    'test_auc': test_auc,
                    'test_acc': test_acc,
                    'total_params': total_params,
            }
                      
            # wandb.log(metric_per_iter)
            if epoch%args.save_after_ep == 0:
                model_save_folder = args.model_folder+exp_name+'/'
                if not os.path.exists(model_save_folder):
                    os.makedirs(model_save_folder)
                model_name = 'model_'+str(epoch)+'.tar'
                print(model_name)
                torch.save(model.state_dict(), model_save_folder+model_name)
                is_best = val_auc > best_auc
                best_auc = max(val_auc, best_auc)
                is_best = val_auc > best_val_auc
                best_auc = max(val_auc, best_val_auc)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    final_test_auc = test_auc
                    ep = epoch

        print(f'Final Test: {final_test_auc:.4f}', ep, best_val_auc)
        best_model = model_save_folder+'model_'+str(ep)+'.tar'
        
        print('best model:',best_model)
        
        # Append the path of the best model to best_models.txt
        with open(args.best_models_file, 'a') as file:
            file.write(best_model + '\n')

    elif args.train_flag=='0':
        with open(args.best_models_file, 'r') as file:
            lines = file.readlines()
            model_file = lines[args.line_num].strip()  # Adjusting for 0-based indexing

        model.load_state_dict(torch.load(model_file))
        parts = model_file.split('/')
        model_name = parts[-2] + parts[-1].split('tar')[0]                  
        
        print('Train:')
        train_loss,train_auc, train_acc, train_given, train_preds, train_plabels, tr_sp, tr_ch, tr_efts = predict_from_model(train_graph, train_loader, model, criterion, device)
        print('Val:')
        val_loss,val_auc, val_acc, val_given, val_preds, val_plabels, v_sp, v_ch , v_efts = predict_from_model(train_graph, val_loader, model, criterion, device)
        print('Test:')
        test_loss,test_auc, test_acc, test_given, test_preds, test_plabels, ts_sp, ts_ch, ts_efts = predict_from_model(train_graph, test_loader, model, criterion, device)
        np.savetxt(args.results_folder+'train/'+model_name+'_train.csv', np.column_stack((np.array(tr_sp), np.array(tr_ch), np.array(tr_efts), np.array(train_given), np.array(train_preds))), delimiter=',')
        np.savetxt(args.results_folder+'val/'+model_name+'_val.csv', np.column_stack((np.array(v_sp), np.array(v_ch), np.array(v_efts), np.array(val_given), np.array(val_preds))), delimiter=',')
        np.savetxt(args.results_folder+'test/'+model_name+'_test.csv', np.column_stack((np.array(ts_sp), np.array(ts_ch), np.array(ts_efts), np.array(test_given), np.array(test_preds))), delimiter=',')
        print('Train','roc_auc_score', 'f1_score', 'precision_score', 'recall_score')
        print('Train',roc_auc_score(train_given,train_preds), f1_score(train_given, train_plabels), precision_score(train_given, train_plabels), recall_score(train_given, train_plabels))
        print('Val',roc_auc_score(val_given,val_preds), f1_score(val_given, val_plabels), precision_score(val_given, val_plabels), recall_score(val_given, val_plabels))
        print('Test',roc_auc_score(test_given,test_preds), f1_score(test_given, test_plabels), precision_score(test_given, test_plabels), recall_score(test_given, test_plabels))

