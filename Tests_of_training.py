# Esse projeto tem como dependências códigos de terceiros com livre distribuição:
#
# python-numpy - Para integração com algebra linear
# python-scipy - integração e otimização numérica
# python-matplotlib - Plotagens
# ipython - parallel and distributed computing (SPMD/MPMD)**
# python-pandas - Analise e modelagem de dados
# python-sympy - algebra**
# pybrain - Rotinas para criações de redes neorais artificiais e treinamentos

# Criando rede neoral
from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2, 3, 1, bias=False)

# Construindo tipo de entrada
from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(2, 1)

#       Imputando valores para serem reconhecidos (XOR)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

# Treinando
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds)

#Iniciando
trainer.train()

net.activate([2, 1])
