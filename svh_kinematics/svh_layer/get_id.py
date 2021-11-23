import numpy as np
a = list(np.load('idx_list.npy'))
print(a)
b = np.load('idxs.npy')
a.insert(0, 0)
l = []
s = 0
for i in range(23):
    mask = np.logical_and(b>=a[i], b<a[i+1])
    idxs = b[mask] - a[i]
    s += len(idxs)
    print(s, end=' ')
    # np.save('vert_ids_{}.npy'.format(i), idxs)

# 726 1475 1745 1834 2004 2098 2156 2186 2273 2323 2359 2386 2481 2538 2573 2602 2701 2758 2796 2826 2925 2965 3000
