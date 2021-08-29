import torch
import matplotlib.pyplot as plt
import torch.nn.functional as function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model :
    def __init__(self) :
        self.W = torch.tensor([[0.0]], requires_grad=True, device=device)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True, device=device)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return function.mse_loss(self.f(x), y)  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability

def performLinearRegression(dataList) :
    epoch_amount = 1000000
    step_size = 0.0001

    print("performing two-dimensional linear regression using "+ device.type)
    print("with " + str(epoch_amount) + " epochs, and a step size of: " + str(step_size))
    model = Model()
    headers = dataList.pop(0)
    x_data = []
    y_data = []
    for row in dataList:
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
 
    x = torch.tensor(x_data,dtype=torch.float32, device=device).reshape(-1,1)
    y = torch.tensor(y_data,dtype=torch.float32, device=device).reshape(-1,1)


    optimizer = torch.optim.SGD([model.W,model.b],step_size)

    
    frac = 100/epoch_amount
    current = 0

    for epoch in range(epoch_amount) :
        model.loss(x,y).backward()
        optimizer.step()
        optimizer.zero_grad()
        current+=1
        print("  ",int(current*frac), "%", end='\r')

    x = x.to("cpu")
    y = y.to("cpu")
    model.W = model.W.to("cpu")
    model.b = model.b.to("cpu")


    print("W = %s, b = %s, loss = %s" % (model.W[0].item(), model.b[0].item(), model.loss(x, y).item()))

    plt.plot(x, y, 'o', label='$(x^{(i)},y^{(i)})$')
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    x1 = torch.tensor([[torch.min(x)], [torch.max(x)]])
    plt.plot(x1, model.f(x1).detach(), label='$\\hat y = f(x) = xW+b$')
    plt.legend()
    plt.show()