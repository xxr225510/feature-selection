class hhhh:
    def demo(self):
        import numpy as np
        signals=[[1,2,3,4],[5,6,7,8],[9,8,7,6],[5,4,3,2]]
        signal=np.array(signals,dtype=int)
        print signals[0:2,:]
        print signals[1:2,:]
        print signals[2:2,:]
hhhh().demo()