import utils
import numpy as np
from model import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.model_selection import train_test_split,StratifiedKFold
import torch.utils.data as Data

np.random.seed(123)
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

Head = [2]
mean = [0.60]
result = []
for h in Head:
    for m in mean:
        # Parameters
        # ==================================================
        parser = ArgumentParser("MCA-GCN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
        parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
        parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
        parser.add_argument("--dropout", default=0.2, type=float, help="Dropout")
        parser.add_argument("--hidden_size", default=90, type=int)
        parser.add_argument("--nhead", default=h, type=int)
        args = parser.parse_args()
        print(args)
        # Load data
        print('loading data')
        adj_matrixs,roi_features,gene_features,node_features,labels = utils.load_data(m)
        print("Loading data... finished!")
        num_classes = 2

        # Split data into 80% training and 20% test set
        train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels)
        # Get the data and labels for training and test sets
        remain_adj_matrixs = adj_matrixs[train_indices]
        remain_roi_features = roi_features[train_indices]
        remain_gene_features = gene_features[train_indices]
        remain_node_features = node_features[train_indices]
        remain_labels = labels[train_indices]

        test_adj_matrixs = adj_matrixs[test_indices]
        test_roi_features = roi_features[test_indices]
        test_gene_features = gene_features[test_indices]
        test_node_features = node_features[test_indices]
        test_labels = labels[test_indices]

        kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=123)
        ACClist, SENlist, SPElist, BAClist, F1list, MCClist, AUClist, epoch_list = [], [], [], [], [], [], [], []
        scores = []

        for train_index, test_index in kfold.split(remain_node_features,remain_labels):

            train_ACC, train_SEN, train_SPE, train_BAC, train_F1, train_MCC, train_AUC = 0, 0, 0, 0, 0, 0, 0
            test_ACC, test_SEN, test_SPE, test_BAC, test_F1, test_MCC, train_AUC = 0, 0, 0, 0, 0, 0, 0
            ACC, SEN, SPE, BAC, F1, MCC, AUC = 0, 0, 0, 0, 0, 0, 0

            model = MCA_GCN(feat_dim = 90,
                            hidden_size=args.hidden_size,
                            num_classes=num_classes,
                            nhead=args.nhead,
                            dropout=args.dropout).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            loss_func = nn.CrossEntropyLoss().to(device)

            train_adj_matrixs = remain_adj_matrixs[train_index]
            train_roi_features = remain_roi_features[train_index]
            train_gene_features = remain_gene_features[train_index]
            train_node_features = remain_node_features[train_index]
            train_labels = remain_labels[train_index]

            val_adj_matrixs = remain_adj_matrixs[test_index]
            val_roi_features = remain_roi_features[test_index]
            val_gene_features = remain_gene_features[test_index]
            val_node_features = remain_node_features[test_index]
            val_labels = remain_labels[test_index]

            train_dataset = Data.TensorDataset(train_adj_matrixs, train_roi_features, train_gene_features, train_node_features, train_labels)
            val_dataset = Data.TensorDataset(val_adj_matrixs, val_roi_features, val_gene_features, val_node_features, val_labels)
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
            val_loader = Data.DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=True)
            EPOCH = args.num_epochs
            for epoch in range(1, args.num_epochs + 1):
                total_loss = 0.
                #train
                model.train()
                for step, (train_adj_matrixs, train_roi_features, train_gene_features, train_node_features, train_labels) in enumerate(train_loader):
                    train_adj_matrixs = train_adj_matrixs.to(device)
                    train_roi_features = train_roi_features.to(device)
                    train_gene_features = train_gene_features.to(device)
                    train_node_features = train_node_features.to(device)
                    train_labels = train_labels.to(device)
                    output,normalized_node_scores = model(train_roi_features,train_gene_features,train_adj_matrixs)

                    loss = loss_func(output, train_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    train_ACC, train_SEN, train_SPE, train_BAC, train_F1, train_MCC, train_AUC = utils.stastic_indicators(output,train_labels)

                #val
                model.eval()
                total_test_loss = 0.
                with torch.no_grad():
                    for step, (val_adj_matrixs, val_roi_features, val_gene_features, val_node_features, val_labels) in enumerate(val_loader):
                        val_adj_matrixs = val_adj_matrixs.to(device)
                        val_roi_features = val_roi_features.to(device)
                        val_gene_features = val_gene_features.to(device)
                        val_node_features = val_node_features.to(device)
                        val_labels = val_labels.to(device)
                        outputs,normalized_node_scores = model(val_roi_features,val_gene_features,val_adj_matrixs)
                        val_loss = loss_func(outputs, val_labels)
                        total_val_loss = total_test_loss + val_loss.item()
                        ACC, SEN, SPE, BAC, F1, MCC, AUC = utils.stastic_indicators(outputs,val_labels)
                        print('| epoch {:3d} | test acc {:5.2f}  | sen {:5.2f} | spe {:5.2f} | auc {:5.2f} | F1 {:5.2f} | mcc {:5.2f} | auc {:5.2f}'.format(
                                epoch, ACC * 100, SEN * 100, SPE * 100, BAC * 100, F1 * 100, MCC * 100, AUC * 100))
                        if test_ACC < ACC:
                            test_adj_matrixs = test_adj_matrixs.to(device)
                            test_roi_features = test_roi_features.to(device)
                            test_gene_features = test_gene_features.to(device)
                            test_node_features = test_node_features.to(device)
                            test_labels = test_labels.to(device)
                            outputs_test, normalized_node_scores = model(test_roi_features, test_gene_features,test_adj_matrixs)
                            test_loss = loss_func(outputs_test, test_labels)
                            ACC, SEN, SPE, BAC, F1, MCC, AUC = utils.stastic_indicators(outputs_test, test_labels)

                            test_ACC = ACC
                            test_SEN = SEN
                            test_SPE = SPE
                            test_BAC = BAC
                            test_F1 = F1
                            test_MCC = MCC
                            test_AUC = AUC
                            test_epoch = epoch
                            scores.append(normalized_node_scores)
            print('| epoch {:3d}  | test_loss {:5.2f}  | test acc {:5.2f}  | sen {:5.2f} | spe {:5.2f} | bac {:5.2f} | F1 {:5.2f} | mcc {:5.2f} | auc {:5.2f}'.format(
                    test_epoch, total_val_loss, test_ACC * 100, test_SEN * 100, test_SPE * 100, test_BAC * 100, test_F1 * 100, test_MCC * 100, test_AUC * 100))
            ACClist.append(test_ACC)
            SENlist.append(test_SEN)
            SPElist.append(test_SPE)
            BAClist.append(test_BAC)
            F1list.append(test_F1)
            MCClist.append(test_MCC)
            AUClist.append(test_AUC)
            epoch_list.append(test_epoch)

        print(ACClist)
        print(SENlist)
        print(SPElist)
        print(BAClist)
        print(F1list)
        print(MCClist)
        print(AUClist)
        print(epoch_list)
        print('acc_mean:{}'.format(sum(ACClist)/len(ACClist)))
        print('sen_mean:{}'.format(sum(SENlist)/len(SENlist)))
        print('spe_mean:{}'.format(sum(SPElist)/len(SPElist)))
        print('bac_mean:{}'.format(sum(BAClist)/len(BAClist)))
        print('F1_mean:{}'.format(sum(F1list)/len(F1list)))
        print('mcc_mean:{}'.format(sum(MCClist)/len(MCClist)))
        print('auc_mean:{}'.format(sum(AUClist)/len(AUClist)))

        scores = torch.stack(scores)
        mean_score = scores.mean(dim=0)
        mean_score = mean_score.cpu().numpy()
        file_path = '../data/score.txt'
        numpy_array_column = mean_score.reshape(-1, 1)
        np.savetxt(file_path, numpy_array_column, delimiter=',', fmt='%.6f')
        acc_result = sum(ACClist) / len(ACClist)
        result.append(acc_result)
    result.append('***')

