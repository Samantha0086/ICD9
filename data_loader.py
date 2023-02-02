def data_loader(train_inputs,train_inputs2,  val_inputs,val_inputs2, train_labels, val_labels,
                batch_size):
    """Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """
    # train_inputs = np.concatenate((train_inputs, train_inputs2), axis=1)
    # val_inputs = np.concatenate((val_inputs, val_inputs2), axis=1)
    # Convert data type to torch.Tensor
    train_inputs, train_inputs2, val_inputs, val_inputs2, train_labels, val_labels =\
    tuple(torch.tensor(data) for data in
          [train_inputs, train_inputs2.astype(np.float32), val_inputs, val_inputs2.astype(np.float32), train_labels, val_labels])



    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_inputs2, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs,val_inputs2, val_labels)
    val_sampler = SequentialSampler(val_data)
    
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader



