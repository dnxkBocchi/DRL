# def timeReward2(self, state, action):
#     budget = state[0]
#     deadline = state[1]
#     bft = state[2]
#     lft = state[3]
#     time = state[6:12]
#     cost = state[12:18]

#     if time[action]<=lft:
#         if deadline != min(time):
#             time_r = (lft - time[action])/(lft - min(time))
#             time_r = time_r.item()
#         else:
#             time_r = 1 #(min(time) - time[action])/(max(time) - min(time))
#     else:
#         if max(time)!=lft:
#             time_r = (lft - time[action])/(max(time) - lft)
#             time_r = time_r.item()
#         else:
#             time_r = -1#(max(time) - time[action])/(max(time) - min(time))

#     return time_r

# def reward3(self, state, action, task):
#     budget = state[0]
#     deadline = state[1]
#     bft = state[2]
#     lft = state[3]
#     time = state[6:12]
#     cost = state[12:18]
#     t = time[action]
#     c = cost[action]

#     reward_array = [0]*6
#     cidx = sorted(range(6), key=lambda k: cost[k]) #, reverse=True)

#     zd = []
#     nd = []
#     znd = []
#     d = []


#     for i in cidx:
#         if cost[i] == 0:
#             if time[i]<=deadline or (time[i]>deadline and time[i]<=bft):
#                 zd.append(i);
#             else:
#                 znd.append(i);
#         else:
#             if time[i]<=deadline or (time[i]>deadline and time[i]<=bft):
#                 d.append(i);
#             else:
#                 nd.append(i);

#     a = 0
#     b = 0
#     c = 0

#     zd.sort(key=lambda k:time[k])
#     for i in zd:
#         reward_array[i] = 0.5*(1 - 1/6 * a) + 0.5
#         # if (i==action): print("ZD")
#         a += 1


#     #==========================================
#     b = a + 0
#     bsort = sorted(d, key=lambda k:cost[k])
#     dsort = sorted(d, key=lambda k:time[k])
#     for i in dsort:
#         reward_array[i] = 0.5*(1 - 1/6 * b)
#         b += 1

#     b = a + 0
#     for i in bsort:
#         # if (i==action): print("D")
#         reward_array[i] = reward_array[i] + 0.5*(-1/6 * b)
#         b += 1

#     #==========================================

#     c = 1
#     znd.sort(key=lambda k:time[k])
#     for i in znd:
#         reward_array[i] = 0.5*(-1/6 * c)+0
#         # if (i==action): print("ZND")
#         c += 1

#     #==========================================

#     d = c + 0
#     bsort = sorted(nd, key=lambda k:cost[k])
#     dsort = sorted(nd, key=lambda k:time[k])
#     for i in dsort:
#         reward_array[i] = 0.5*(-1/6 * d)
#         d += 1

#     d = c + 0
#     for i in bsort:
#         # if (i==action): print("ND")
#         reward_array[i] = reward_array[i] + 0.5*(-1/6 * d)
#         d += 1

#     r = reward_array[action]
#     return r

# def reward4(self, state, action, task):
#     budget = state[0]
#     deadline = state[1]
#     bft = state[2]
#     lft = state[3]
#     time = state[6:12]
#     cost = state[12:18]
#     t = time[action]
#     c = cost[action]

#     reward_array = [0]*6
#     idx = sorted(range(6), key=lambda k: time[k]) #, reverse=True)

#     zl = []
#     bftl = []
#     lftl = []
#     dl = []

#     counter = 0
#     if bft <= deadline:
#         if lft <= deadline:
#             for i in idx:
#                 if time[i] <= bft:
#                     zl.append(i)
#                 elif time[i] <= lft:
#                     bftl.append(i)
#                 elif time[i] <= deadline:
#                     lftl.append(i)
#                 else:
#                     dl.append(i)

#             for idx in zl[::-1]:
#                 reward_array[idx] = 1 - 1/6 * counter
#                 counter += 1

#             for idx in bftl:
#                 reward_array[idx] = 1 - 1/6 * counter
#                 counter += 1

#             counter = 0
#             for idx in lftl:
#                 reward_array[idx] = 0 #-0.1 * counter
#                 counter += 1

#             counter = 1
#             for idx in dl:
#                 reward_array[idx] = -1/6 * counter
#                 counter += 1

#         else:
#             for i in idx:
#                 if time[i] <= bft:
#                     zl.append(i)
#                 elif time[i] <= deadline:
#                     bftl.append(i)
#                 elif time[i] <= lft:
#                     dl.append(i)
#                 else:
#                     lftl.append(i)

#             for idx in zl[::-1]:
#                 reward_array[idx] = 1 - 1/6 * counter
#                 counter += 1

#             for idx in bftl:
#                 reward_array[idx] = 1 - 1/6 * counter
#                 counter += 1

#             counter = 0
#             for idx in dl:
#                 reward_array[idx] = -1/6 * counter
#                 counter += 1

#             for idx in lftl:
#                 reward_array[idx] = -1/6 * counter
#                 counter += 1
#     else:
#         for i in idx:
#             if time[i] <= deadline:
#                 zl.append(i)
#             elif time[i] <= bft:
#                 dl.append(i)
#             elif time[i] <= lft:
#                 bftl.append(i)
#             else:
#                 lftl.append(i)

#         for idx in dl[::-1]:
#             reward_array[idx] = 1 - 1/6 * counter
#             counter += 1

#         for idx in zl[::-1]:
#                 reward_array[idx] = 1 - 1/6 * counter
#                 counter += 1

#         counter = 0
#         for idx in bftl:
#             reward_array[idx] = -1/6 * counter
#             counter += 1

#         for idx in lftl:
#             reward_array[idx] = -1/6 * counter
#             counter += 1


#     cidx = sorted(range(6), key=lambda k: time[k])
#     counter = 1
#     for ci in cidx:
#         if reward_array[ci]<0:
#             reward_array[ci]*=0.5
#         elif cost[ci] == 0:
#             reward_array[ci] = reward_array[ci]*0.5 + 0.5
#         else:
#             reward_array[ci] = reward_array[ci]*0.5 +  0.5*counter*-1/6
#         counter += 1

#     r = reward_array[action]
#     return r
