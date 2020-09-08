import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt


class TitanicDataset_train(Dataset):  # 训练集
    def __init__(self, filepath):
        xy = pd.read_csv(filepath)
        xy = xy[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        xy["Sex"] = [1 if x == "male" else 0 for x in xy["Sex"]]
        xy["Age"] = xy["Age"].fillna(xy["Age"].mean())
        y_Data = xy["Survived"]
        y_Data = np.array(y_Data, dtype=np.float32).reshape(-1, 1)
        f = xy["Embarked"].value_counts()
        f1 = f["S"] / (f["S"] + f["C"] + f["Q"])
        f2 = f["C"] / (f["S"] + f["C"] + f["Q"])
        f3 = f["Q"] / (f["S"] + f["C"] + f["Q"])
        temp = []
        for i in xy["Embarked"]:
            if i == "S":
                temp.append(f1)
            elif i == "C":
                temp.append(f2)
            elif i == "Q":
                temp.append(f3)
        np.asarray(temp).reshape(-1, 1)

        xy["Embarked"] = temp
        x_Data = xy[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        x_Data = np.array(x_Data, dtype=np.float32)
        self.y_Data = torch.from_numpy(y_Data)
        self.x_Data = torch.from_numpy(x_Data)
        self.len = xy.shape[0]

    def __getitem__(self, item):
        return self.x_Data[item], self.y_Data[item]

    def __len__(self):
        return self.len


class TitanicDatatest_Test(Dataset):  #  测试集
    def __init__(self, filepath):
        x = pd.read_csv(filepath)
        x = x[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        x["Sex"] = [0 if i == "male" else 1 for i in x["Sex"]]
        x["Age"] = x["Age"].fillna(x["Age"].mean())
        f = x["Embarked"].value_counts()
        f1 = f["S"] / (f["S"] + f["C"] + f["Q"])
        f2 = f["C"] / (f["S"] + f["C"] + f["Q"])
        f3 = f["Q"] / (f["S"] + f["C"] + f["Q"])
        temp = []
        for i in x["Embarked"]:
            if i == "S":
                temp.append(f1)
            elif i == "C":
                temp.append(f2)
            elif i == "Q":
                temp.append(f3)
        np.asarray(temp).reshape(-1, 1)
        x["Embarked"] = temp
        x_Data = np.array(x, dtype=np.float32)
        self.x_Data = torch.from_numpy(x_Data)
        self.len = x_Data.shape[0]

    def __getitem__(self, item):
        return self.x_Data[item]

    def __len__(self):
        return self.len


train_dataset = TitanicDataset_train("titanic/train.csv")
train_loader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=2, shuffle=True)
test_dataset = TitanicDatatest_Test("titanic/test.csv")
test_loader = DataLoader(dataset=test_dataset, num_workers=2, shuffle=False)


class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 5)
        self.linear2 = torch.nn.Linear(5, 3)
        self.linear3 = torch.nn.Linear(3, 2)
        self.linear4 = torch.nn.Linear(2, 1)

    def forward(self, x):
        z = F.relu(self.linear1(x))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        y_pred = torch.sigmoid(self.linear4(z))
        return y_pred


model = model()
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lost_list = []

def test():
    y_out = []
    with torch.no_grad():
        for data in test_loader:
            y_pred = model(data)
            y_out.append(y_pred.item())
    return y_out




if __name__ == '__main__':
    for epoch in range(100):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, label = data
            y_pred = model(inputs)
            loss = criterion(y_pred, label)
            print("epoch:{},batch:{},loss:{}".format(epoch + 1, i + 1, loss.item()))
            running_loss += loss
            if i == 27:
                average_loss = running_loss / 28
                lost_list.append(average_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    y_out = test()
    y_out = [1 if i > 0.5 else 0 for i in y_out]
    real_output = pd.read_csv("titanic/gender_submission.csv")
    y_out = np.array(y_out).reshape(-1,1)
    real_output["result"] = np.zeros([418,1])
    result = []
    for i in range(y_out.shape[0]):
        if y_out[i] == real_output["Survived"][i]:
            result.append("True")
        else:
            result.append("False")
    frequency = pd.value_counts(result)
    T_fre = frequency["True"]/(frequency["True"]+frequency["False"])
    F_fre = frequency["False"] / (frequency["True"] + frequency["False"])
    print("True possibility:{},False possibility:{}".format(T_fre,F_fre))