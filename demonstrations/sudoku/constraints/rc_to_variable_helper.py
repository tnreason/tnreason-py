
def rc_to_pos_assignment(r,c):
    r1 = r // 3
    r2 = r % 3
    c1 = c // 3
    c2 = c % 3
    return "pos_" + str(r1) + "_" + str(r2) + "_" + str(c1) + "_" + str(c2)