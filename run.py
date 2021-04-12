import gnn_fw as gf
from random import random
import torch
from torch_geometric.data import DataLoader
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset = gf.utils.COVIDDataset(root='./')
final_mape_dict = dict()

nfolds = 5
number_of_epochs = 10

for fold in range(nfolds):
    total_dataset = dataset[:298]
    test_dataset = dataset[298:]
    # total_dataset = sorted(total_dataset, key=lambda x: random())
    train_dataset = total_dataset[:200]
    val_dataset = total_dataset[200:]

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore")
    model_gcn = gf.models.GCN()
    print(model_gcn)
    print("Number of parameters: ", sum(p.numel() for p in model_gcn.parameters()))

    optimizer = torch.optim.Adam(model_gcn.parameters(), lr=0.007)

    # Wrap data in a data loader
    NUM_GRAPHS_PER_BATCH = 1
    loader = gf.utils.data_loader(dataset=train_dataset, batch_size=NUM_GRAPHS_PER_BATCH)
    val_loader = gf.utils.data_loader(dataset=val_dataset, batch_size=NUM_GRAPHS_PER_BATCH)
    test_loader = gf.utils.data_loader(dataset=test_dataset, batch_size=NUM_GRAPHS_PER_BATCH)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200,
                                                           threshold=1e-6)

    print("Starting training and validating...")
    best_loss = 0.5
    losses_tr = []
    losses_vl = []
    for epoch in range(number_of_epochs):
        # training on the same epoch
        loss_tr, h_tr = gf.models.training(data_loader=loader, model=model_gcn)
        losses_tr.append(loss_tr)
        # validation on the same epoch
        loss_vl, h_vl = gf.models.validation(data_loader=val_loader, model=model_gcn)
        losses_vl.append(loss_vl)

        scheduler.step(loss_tr)

        if loss_vl < best_loss:
            best_loss = loss_vl
            checkpoint_path = f"training_2/cp-{epoch:04d}.ckpt"
            model_save_path = os.path.dirname(checkpoint_path)
            torch.save(model_gcn.state_dict(), model_save_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss {loss_tr}")


    # Visualize learning
    def plot_losses(training_losses, validation_losses):
        fig, axs = plt.subplots(ncols=2)
        for num, losses in enumerate[training_losses, validation_losses]:
            losses_float = [float(loss_.cpu().detach().numpy()) for loss_ in losses]
            loss_indices = [i for i, l in enumerate(losses_float)]
            axs[num] = sns.lineplot(loss_indices, losses_float)
        plt.savefig(f"Fold_{fold}_losses.jpg")


    plot_losses(training_losses=losses_tr, validation_losses=losses_vl)

    # Analyze the results for one batch
    for batch in test_loader:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            batch.to(device)
            model_gcn = torch.load(model_save_path)
            pred, embed = model_gcn(batch.x.float(), batch.edge_index, batch.batch)
            print(batch.y)
            print(pred)
            MAPE = gf.utils.mean_absolute_percentage_error(y_true=batch.y.cpu().numpy(), y_pred=pred.cpu().numpy())
            print(MAPE)
            final_mape_dict[fold] = MAPE

average_mape = np.mean(list(final_mape_dict.values()))
std_mape = np.std(list(final_mape_dict.values()))
print(f"Final results i.e., Mean+SD MAPE: {average_mape} +/- {std_mape}")
