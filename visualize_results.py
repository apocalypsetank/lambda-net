import scipy.io as sio
import matplotlib.pyplot as plt

for i in range(10):
    print i+1
    img=sio.loadmat('result/20190901-155700/test_result'+str(i+1)+'.mat')['result']
    plt.imshow(img[:,:,10])
    plt.show()

