# Import ##################################################################################
from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import easydict
import numpy as np
from logger import Logger

torch.manual_seed(0)

# Monkey Patching #########################################################################
def step3(self, closure=None):
    """Performs a single optimization step.
    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)

            # if momentum != 0:
            #     param_state = self.state[p]
            #     if 'momentum_buffer' not in param_state:
            #         buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            #         buf.mul_(momentum).add_(d_p)
            #     else:
            #         buf = param_state['momentum_buffer']
            #         # buf.mul_(momentum).add_(1 - dampening, d_p)
            #         # buf.mul_(momentum).add_(1 - dampening, group['lr'] * d_p)
            #         buf.mul_(momentum).add_(1 - dampening, d_p)
            #         # buf.mul_(momentum).add_(1 - dampening, d_p*torch.pow(c*0.1,0.5)/group['lr'])
            #         # buf.mul_(momentum).add_(1 - dampening, d_p)
            #         # buf.mul_(momentum.mul_(a*b)).add_(1 - dampening, group['lr']*d_p)
            #     if nesterov:
            #         d_p = d_p.add(momentum, buf)
            #     else:
            #         d_p = buf

            # if momentum != 0:
            #     param_state = self.state[p]
            #     if 'momentum_buffer' not in param_state:
            #         buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            #         buf.mul_(momentum).add_(d_p)
            #         ccc = param_state['count'] = torch.zeros_like(p.data)
            #         ccc.add_(0)
            #         c = param_state['drift'] = torch.zeros_like(p.data)
            #         c.add_(0)
            #         #     c.add_(1.0)
            #         # ccc.add_(1)
            #         # print(ccc)
            #         # print(d_p.size())
            #         # print(np.sum(buf.numpy()))
            #     else:
            #         buf = param_state['momentum_buffer']
            #         ccc = param_state['count']
            #         c = param_state['drift']
            #
            #         # print(ccc)
            #         # print(np.size(buf.numpy()))
            #         # print(torch.sign(d_p))
            #         # print(np.sum(d_p.numpy()))
            #         ccc.mul_(0)
            #         ccc[(torch.sign(d_p) > 0) & (torch.sign(buf) > 0)] = 1
            #         c.add_(ccc)
            #         ccc.mul_(0)
            #         ccc[(torch.sign(d_p) < 0) & (torch.sign(buf) < 0)] = -1
            #         c.add_(ccc)
            #         ccc.mul_(0)
            #         ccc.add_(1)
            #         ccc[(torch.sign(d_p) * torch.sign(buf)) != 1] = 0
            #         c.mul_(ccc)
            #         ccc.mul_(0)
            #         ccc.add_(torch.sign(c) * ((torch.pow(torch.abs(c), 0.08)) - 1))
            #         # print(torch.sign(d_p))
            #         # c.mul_(ccc)
            #         # torch.set_printoptions(edgeitems=15,precision=10)
            #         print(ccc)
            #         # ccc.mul_(torch.sign(torch.abs(buf) - torch.abs(d_p)).add_(1).mul_(0.5))
            #         # print(torch.pow(c+1,0.001))
            #         # buf.mul_(momentum).add_(1 - dampening, d_p)
            #         # buf.mul_(momentum).add_(1 - dampening, group['lr'] * d_p)
            #         # buf.mul_(momentum*ccc).add_(1 - dampening, d_p)
            #         buf.mul_(momentum).add_(1 - dampening, d_p + d_p * ccc)
            #         # buf.mul_(momentum * (torch.pow(ccc.mul_(0.0001), 0.01) + 1.0)).add_(1 - dampening, d_p)
            #         # buf.mul_(momentum).add_(1 - dampening, d_p*torch.pow(c*0.1,0.5)/group['lr'])
            #         # buf.mul_(momentum).add_(1 - dampening, d_p)
            #         # buf.mul_(momentum.mul_(a*b)).add_(1 - dampening, group['lr']*d_p)
            #
            #     if nesterov:
            #         d_p = d_p.add(momentum, buf)
            #     else:
            #         d_p = buf

            param_state = self.state[p]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf.mul_(momentum).add_(d_p)
                ccc = param_state['count'] = torch.zeros_like(p.data)
                ccc.add_(0)
                c = param_state['drift'] = torch.zeros_like(p.data)
                c.add_(0)
                #     c.add_(1.0)
                # ccc.add_(1)
                # print(ccc)
                # print(d_p.size())
                # print(np.sum(buf.numpy()))
            else:
                buf = param_state['momentum_buffer']
                ccc = param_state['count']
                c = param_state['drift']

                # print(ccc)
                # print(np.size(buf.numpy()))
                # print(torch.sign(d_p))
                # print(np.sum(d_p.numpy()))
                ccc.mul_(0)
                ccc[(torch.sign(d_p) > 0) & (torch.sign(buf) > 0)] = 1
                c.add_(ccc)
                ccc.mul_(0)
                ccc[(torch.sign(d_p) < 0) & (torch.sign(buf) < 0)] = -1
                c.add_(ccc)
                ccc.mul_(0)
                ccc.add_(1)
                ccc[(torch.sign(d_p) * torch.sign(buf)) != 1] = 0
                c.mul_(ccc)
                # print(c)
                # ccc.mul_(0)
                # # ccc.add_(torch.sign(c) * ((torch.pow(torch.abs(c), 0.5)) - 1))
                # ccc.add_((torch.pow(torch.abs(c), 0.5)) - 1)
                # print(torch.sign(d_p))
                # c.mul_(ccc)
                # torch.set_printoptions(edgeitems=15,precision=10)

                # ccc.mul_(torch.sign(torch.abs(buf) - torch.abs(d_p)).add_(1).mul_(0.5))
                # print(torch.pow(c+1,0.001))
                # buf.mul_(momentum).add_(1 - dampening, d_p)
                # buf.mul_(momentum).add_(1 - dampening, group['lr'] * d_p)
                # buf.mul_(momentum*ccc).add_(1 - dampening, d_p)
                buf.mul_(momentum).add_(1 - dampening, d_p + d_p * torch.pow(torch.abs(c),3))
                # buf.mul_(momentum).add_(1 - dampening, d_p + d_p * c*1.3)
                # buf.mul_(momentum * (torch.pow(ccc.mul_(0.0001), 0.01) + 1.0)).add_(1 - dampening, d_p)
                # buf.mul_(momentum).add_(1 - dampening, d_p*torch.pow(c*0.1,0.5)/group['lr'])
                # buf.mul_(momentum).add_(1 - dampening, d_p)
                # buf.mul_(momentum.mul_(a*b)).add_(1 - dampening, group['lr']*d_p)

            if nesterov:
                d_p = d_p.add(momentum, buf)
            else:
                d_p = buf


            #
            # p.data.add_(-group['lr'], d_p * c)
            p.data.add_(-group['lr'], d_p)
            # p.data.add_(-1, d_p)

    return loss

# Network #################################################################################
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256, bias=False)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.constant_(self.fc1.bias,0)
        #         self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10, bias=False)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        # torch.nn.init.constant_(self.fc3.bias, 0)
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        #         x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

    #     def __init__(self):
    #         super(MLPNet, self).__init__()
    #         self.fc1 = nn.Linear(28*28, 500)
    #         self.fc2 = nn.Linear(500, 256)
    #         self.fc3 = nn.Linear(256, 10)
    #     def forward(self, x):
    #         x = x.view(-1, 28*28)
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = F.softmax(self.fc3(x))
    #         return x
    def name(self):
        return "MLP"

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def name(self):
        return "LeNet"

# Load Dataset ############################################################################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
print('==>>> total training batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

# Training ###################################################################################
def train(args, model, device, train_loader, criterion, optimizer, epoch):
    torch.manual_seed(args.seed)
    model.train()
    train_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            x, target = x.to(device), target.to(device)
        #         x, target = Variable(x), Variable(target)
        out = model(x)
        loss = criterion(out, target)
        #         ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        train_loss += loss.item()/args.train_batch_size

        if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(train_loader):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Train Loss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * (batch_idx + 1)/ len(train_loader), train_loss))
            train_loss = 0.0
        # if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
        #     print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
        #         epoch, batch_idx + 1, ave_loss / 100))
        #     ave_loss = 0.0

# Testing ####################################################################################
def test(args, model, device, criterion, test_loader):
    torch.manual_seed(args.seed)
    model.eval()
    correct_cnt, test_loss = 0, 0
    # total_cnt = 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_loader):
            if use_cuda:
                x, target = x.to(device), target.to(device)
            #         x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x)
            loss = criterion(out, target)
            test_loss += loss.item()/args.test_batch_size
            _, pred_label = torch.max(out.data, 1)
            # total_cnt += target.size(0)
            correct_cnt += (pred_label == target).sum().item()

            # smooth average
            #         ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
            if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(test_loader):
                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                    test_loss, correct_cnt, len(test_loader.dataset),
                    100. * correct_cnt / len(test_loader.dataset)))

                # print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                #     epoch, batch_idx + 1, loss, correct_cnt * 1.0 / total_cnt))

model = MLPNet()
# for a in model.parameters():
#     print(a)
#     print(a.size())
args=easydict.EasyDict({'train_batch_size':100, 'test_batch_size':100, 'epochs':50, 'momentum':0.0, 'lr':0.01, 'no_cuda':False, 'seed':0, 'log_interval':100, 'save_model':False})
if use_cuda:
    model = model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(params=model.parameters(), momentum=args.momentum, lr=args.lr)
optim.SGD.step = step3
criterion = nn.CrossEntropyLoss()

print('\n===> Training starts')
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, criterion, optimizer, epoch)
    test(args, model, device, criterion, test_loader)

if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")