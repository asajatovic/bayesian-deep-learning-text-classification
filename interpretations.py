def grad_score_on_example(x, y, model, criterion):
    # for _ in T: ???
    # batch_size = 1 because of gradient calc
    assert x.ndimension() == 1
    out = model(x).squeeze(1)
    loss = criterion(out, y)
    loss += model.kl_loss()
    loss.backward()

    score = out  # F.softmax(out)
    pred = torch.argmax(score, dim=1)
    # get a list of gradients w.r.t to each word (via its embedding)
    embed_grads = model.embedding.weight.grad * model.embedding.weight
    onehot_grads = embed_grads.sum(dim=1).tolist()

    x_list = x.tolist()
    x_grads = [onehot_grads[i] for i in x_list]
    word_scores = list(zip(x_list, x_grads))  # list of (word, word_grad)

    # return pred, score, onehot_grads
    return word_scores


def leave_one_out_on_example(x, model):
    def flatten(x):
        """generate a batch of examples, x, each entry with a different word left out"""
        n_words = x.shape[0]
        assert x.ndimension() == 1
        assert n_words > 1
        xs = []
        for i in range(n_words):
            # what about padding? - nothing :)
            xs.append(torch.cat((x[:i], x[i+1:]), dim=0))
        assert len(xs) == n_words
        return torch.stack(xs)

    assert x.ndimension() == 1
    y_pred = predict_on_batch(x, model, grad=False, inference=False)
    # torch.max https://github.com/Eric-Wallace/deep-knn/blob/master/run_dknn.py#L236

    xs = flatten(x)  # batch of leave out one word
    ys = predict_on_batch(xs, model, grad=False, inference=False)

    x_list = x.tolist()
    x_drops = (ys - y_pred).tolist()  # drop in score
    word_scores = list(zip(x_list, x_drops))  # list of (word, word_score)

    return word_scores
