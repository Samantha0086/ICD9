def evaluate(model, val_dataloader, test_loader_sg):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    val_auroc = []

    # For each batch in our validation set...
    for item1, item2 in zip(test_loader_sg, val_dataloader):
        x_batch = item1[:,0]
        y_batch = item1[:,1]
        X, X1,y = item2

        # Compute logits
        with torch.no_grad():
            loss_word2vec, logits = word2vec(x_batch, y_batch, X, X1)

        loss_cnn = criterion_cnn(logits,y)
        loss = (1-weight_cnn)*loss_word2vec +weight_cnn* loss_cnn  # * weight
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        proba = logits[:, 1].detach().numpy()

        # Calculate the accuracy rate
        accuracy = (preds == y).cpu().numpy().mean() * 100
        
        auroc =  roc_auc_score(y, proba)
        

        val_auroc.append(auroc)
        
      
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    

    return val_loss, val_accuracy, val_auroc




