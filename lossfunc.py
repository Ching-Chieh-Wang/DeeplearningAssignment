import numpy as np
# =============================================================================
# class CrossEntropyLoss():
#     def loss( y, p):
#         p = np.clip(p, 1e-15, 1 - 1e-15)
#         return np.sum(-1*( y * np.log(p)+(1-y)*np.log(1-p)))
#     def gradient( y, p):
#         p = np.clip(p, 1e-15, 1 - 1e-15)
#         return - (y / p) +(1-y)/(1-p)+y+(1-y)*-1 #不知道為什麼要加上y+(1-y)*-1才會對
# =============================================================================

class CrossEntropyLoss():
    def loss( y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.sum(-1*( y * np.log(p)))
    def gradient( y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) 

# =============================================================================
# class CrossEntropyLoss():
#     def loss( y, p):
#         p = np.clip(p, 1e-15, 1 - 1e-15)
#         return np.sum(-1*( y * np.log(p)+(1-y)*np.log(1-p)))
#     def gradient( y, p):
#         p = np.clip(p, 1e-15, 1 - 1e-15)
#         return - (y / p) +(1-y)/(1-p)
# =============================================================================
