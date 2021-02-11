import numpy as np
box = [40]*4 + [60]*5 + [75]*6
print(f"A box has contents {box}")
print(f"A box has {len(box)} light bulbs")

two_where_75 = 0
all_three_same = 0
one_of_each = 0
n = 100000

for i in range(n):
    bulbs = sorted(np.random.choice(box, 3, replace=False))
    # Part a
    # Count when 2types are 75W
    num_75 = 0
    for bulb in bulbs:
        if bulb == 75:
            num_75 += 1
    if num_75 == 2:
        two_where_75 += 1
        

    # Part b   
    # All three are the same
    if bulbs == [40,40,40]:
        all_three_same += 1
    elif bulbs == [60,60,60]:
        all_three_same += 1
    elif bulbs == [75,75,75]:
        all_three_same += 1

    # Part c
    # One of each
    if bulbs == [40,60,75]:
        one_of_each += 1
    
print("Probability that two are 75W: %.3f" % (two_where_75/n),)
print("Probability that all three are the same: %.3f" % (all_three_same/n),)
print("Probability of one of each types: %.3f" % (one_of_each/n),)



# Part d
less_than_6 = 0
n = 100000

for i in range(n):
    bulbs = np.random.choice(box, 5, replace=False)
    if 75 in bulbs:
        less_than_6 += 1
        
print("Probability that it takes at least 6 selections to get a 75-W bulb is : %0.3f" % ((1.0 - less_than_6/n)))



