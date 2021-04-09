import torch as t


class Perceptron:
    def __init__(self, input_dim, activator):
        self.W = t.zeros(input_dim, dtype=t.float32)
        self.b = t.tensor([1], dtype=t.float32)
        self.activator = activator

    def __str__(self):
        return f'Weights: {self.W}\n' \
               f'Bias: {self.b}\n{"=" * 10}'

    def predict(self, input_vector):
        return t.tensor(list(map(self.activator, t.matmul(input_vector, self.W) + self.b)))

    def loss(self, prediction, label):
        return prediction - label

    def _one_iter_(self, input_vector, label, lr):
        grad_W = t.matmul(self.loss(self.predict(input_vector), label), input_vector)
        print(f'grad_W: {grad_W}')
        grad_b = t.matmul(self.loss(self.predict(input_vector), label), t.ones(label.shape))
        print(f'grad_b: {grad_b}')
        delta_W = - grad_W * lr
        delta_b = - grad_b * lr
        self.W = self.W + delta_W
        self.b = self.b + delta_b
        print(self.__str__())
        return self.W, self.b

    def train(self, input_vectors, labels, iterations, lr):
        for i in range(iterations):
            self._one_iter_(input_vectors, labels, lr)


def f(x):
    return 1 if x > 0 else 0


def get_training_dataset():
    input_vectors = t.tensor([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=t.float32)
    print(input_vectors.shape)
    labels = t.tensor([1, 0, 0, 0], dtype=t.float32)
    return input_vectors, labels


def train_and_perceptron():
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vectors, labels = get_training_dataset()
    p.train(input_vectors, labels, 20, 0.1)
    #返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print(and_perception.predict(t.tensor([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=t.float32)))
    print('1 and 1 = %d' % and_perception.predict(t.tensor([[1, 1]], dtype=t.float32)))
    print('0 and 0 = %d' % and_perception.predict(t.tensor([[0, 0]], dtype=t.float32)))
    print('1 and 0 = %d' % and_perception.predict(t.tensor([[1, 0]], dtype=t.float32)))
    print('0 and 1 = %d' % and_perception.predict(t.tensor([[0, 1]], dtype=t.float32)))