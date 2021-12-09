import numpy as np

# a = list(np.load('idx_list.npy'))
# print(a)
# b = np.load('idxs.npy')
# a.insert(0, 0)
# l = [0]
# s = 0
# for i in range(22):
#     mask = np.logical_and(b >= a[i], b < a[i + 1])
#     idxs = b[mask] - a[i]
#     # print(idxs)
#     s += len(idxs)
#     l.append(s)
#     print(s, end=' ')
#     # np.save('vert_ids_{}.npy'.format(i+1), idxs)
# print()
# print(l)
# np.save('idx3000_list.npy', l)
# 957 1322 1438 1659 1778 1857 1905 2022 2090 2135 2177 2305 2384 2433 2471 2606 2687 2735 2780 2891 2953 3000


a = list(np.load('idx3000_list.npy'))

b = np.load('../../dataset/only_touch_idxs.npy')

print(b.shape)

# l = [0]
s = 0
for i in range(22):
    mask = np.logical_and(b >= a[i], b < a[i + 1])
    idxs = b[mask] - a[i]
    # print(idxs)
    s += len(idxs)
    # l.append(s)
    print(s, end=' ')
    # np.save('vert_ids_{}.npy'.format(i+1), idxs)
# print()
# print(l)
# np.save('idx3000_list.npy', l)