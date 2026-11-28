# Import local decoder-only tranformer.
# Contains 
from SLM import SmallLanguageModel
import torch


def estimate_val_loss(m):
    m.eval()
    xb, yb = get_batch('val')
    logits, loss = model(xb, yb)
    m.train()
    return loss







# Get Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Construct Model
model = SmallLanguageModel()
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)








params = tuple(p for p in model.parameters() if p.requires_grad)
print(sum(len(p) for p in params))

data_shapely = torch.zeros((len(train_data)), device=device)
# Train Model
for steps in range(max_iters):

    # Occasionally evaluate the loss on train and val sets
    if steps % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # Sample
    xb, yb, ix = get_batch_and_indices('train')

    #losses.backward()
    valLoss = estimate_val_loss(model)
    print(valLoss, type(valLoss))

    # Get all trainable model parameters.



    # Evaluate Loss
    logits, loss = model(xb,yb)   

    g_val = torch.autograd.grad(valLoss,params)

    g_tr = torch.autograd.grad(loss,params)

    trImpactOnConvergence = sum((gv*gt).sum() for gv,gt in zip(g_val,g_tr))
    print(f"Training impact on convergence: {trImpactOnConvergence}")

    data_shapely[ix] += -learning_rate *trImpactOnConvergence
    plt.scatter(steps, trImpactOnConvergence)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(xb,yb)  
    loss.backward()
    optimizer.step()
plt.show()