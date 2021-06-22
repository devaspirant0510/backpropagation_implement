"""
author : seungho
github : devaspirant0510
email  : seungho020510@gmail.com
created by seungho on 2021-06-22
"""
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])
y_train = np.array([
    [0.],
    [1.],
    [1.],
    [0.]
])


# 손실 함수
def MSE(s, y):
    return 1 / (2 * len(s)) * np.sum((s - y) ** 2)


# 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 모델
class Model(object):
    def __init__(self, x, y, lr=0.01, epoch=1000):
        """
        레이어 정의
        :param x: input 값
        :param y: output 값
        :param lr: learning rate
        :param epoch: epoch
        """
        self.input = x
        self.output = y
        self.w1 = np.random.randn(2, 10)
        self.b1 = np.random.randn(1, 10)
        self.w2 = np.random.randn(10, 1)
        self.b2 = np.random.randn(1, 1)
        self.lr = lr
        self.epoch = epoch

    # 생성자에서 정의한 레이어로 순전파 연산
    def forward(self, x):
        self.z1 = x @ self.w1 + self.b1
        self.s1 = sigmoid(self.z1)
        self.z2 = self.s1 @ self.w2 + self.b2
        self.s2 = sigmoid(self.z2)
        return self.s2

    def backward(self):
        # 손실함수 미분
        dLoss = self.s2 - self.output
        # 활성화 함수 미분
        dAct2 = self.s2 * (1 - self.s2)
        grad2 = dLoss * dAct2
        # 레이어 2의 w,b 미분
        self.dw2 = self.s1.T @ grad2
        self.db2 = np.sum(grad2, axis=0)
        # 레이어 2의 s1 미분( 레이어 1의 output 값으로 들어가서 미분됨)
        ds2 = grad2 @ self.w2.T
        # 활성화 함수 미분
        dAct1 = self.s1 * (1 - self.s1)
        grad1 = ds2 * dAct1
        # 레이어 1 w,b 미분
        self.dw1 = self.input.T @ grad1
        self.db1 = np.sum(grad1, axis=0)
        self.__update(self.lr)

    # 손실함수로 loss 값 리턴
    def get_los(self):
        if self.s2 is None:
            return None
        return MSE(self.s2, self.output)

    # backpropagation 연산 후 업데이트
    def __update(self, lr):
        self.w1 -= lr * self.dw1
        self.b1 -= lr * self.db1
        self.w2 -= lr * self.dw2
        self.b2 -= lr * self.db2

    # 정확도 측정
    def accuracy(self, s, y):
        s_mask = np.where(s >= 0.5, 1, 0)
        checking = np.sum(s_mask == y)
        return checking / len(s)

    # 학습
    def train(self, log_loss=False, log_step=10, draw_graph=True):
        """

        :param log_loss: 학습시키면서 loss 값을 보여줄건지 결정 True 면 보여주고 False 면 안보여줌
        :param log_step: loss 값을 몇 epoch 돌았을때마다 보여줄건지 지정
        :param draw_graph: 그래프를 보여줄지 말지 여부를 결정
        :return: None
        """
        # 정확도와 loss 값을 저장하여 그래프에 표시
        accuracy_list = []
        loss_list = []
        for i in range(1,self.epoch+1):
            yhat = self.forward(self.input)
            self.backward()
            loss = model.get_los()
            # 1 epoch 마다 loss 값과 accuracy 값을 저장
            acc = model.accuracy(yhat, self.output)
            loss_list.append(loss)
            accuracy_list.append(acc)
            # log_loss 가 True 일때 지정한 스템마다 loss 값 출력
            if log_loss is True and self.epoch % log_step == 0:
                print(f"{i}/{self.epoch}\t\tloss:{loss}\t\taccuracy:{acc}")
        # draw_graph 가 True 일때 loss,acc 그래프 그리기
        if draw_graph:
            fig, ax = plt.subplots()
            ax.plot(loss_list, "--",label="loss")
            ax.set_ylabel("loss")
            ax.legend(loc="lower right")
            # y축 두개로 만듦
            ax2 = ax.twinx()
            ax2.plot(accuracy_list, "r",label="accuracy")
            ax2.set_ylabel("accuracy")
            ax2.legend(loc="upper right")
            plt.show()

    # 입력값을 받아 forward 연산
    def check(self, input_):
        return self.forward(input_)


model = Model(x_train, y_train, lr=0.3, epoch=1000)
model.train(log_loss=True, log_step=100)

print(model.check(np.array([[0., 0.]])))
print(model.check(np.array([[0., 1.]])))
print(model.check(np.array([[1., 0.]])))
print(model.check(np.array([[1., 1.]])))
