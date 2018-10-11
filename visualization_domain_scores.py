import numpy as np
import matplotlib.pyplot as plt
import ipdb

import  torch
all_score = torch.load('reverse_gradientresnet50_full_bilinear_reverse/domain_score_epoch0.pth.tar')
source_score = all_score['source']
target_score = all_score['target']
# plt.style.use('ggplot')

bins = 100 # (torch.min(source_score), torch.max(source_score), 2)
ipdb.set_trace()
plt.hist(source_score, bins = bins, label='source', color='red', normed='True')
plt.hist(target_score, bins = bins, label='target', color='blue', normed='True')
plt.title('distribution of domain scores')
plt.xlabel('domain scores')
plt.ylabel('number of images')
plt.legend()
plt.show()
