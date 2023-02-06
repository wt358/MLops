

def get_work_time(target, q=0.5, L=30, in_a_row = 50):
    q_ = target.abs().quantile(q)
    tmp = target.abs().map(lambda x: 1 if (x <=q_) else 0)

    _sum = np.array([tmp.iloc[i:(i+L)].sum() for i in range(len(tmp))]) 
    not_work = np.where(_sum == L)[0]

    points = np.where(np.diff(not_work) != 1)[0]+1

    not_work_set = [0]
    start = 0
    for i in range(len(points)+1):
        if i == len(points):
            s = not_work[points[i-1]:][0]
            e = not_work[points[i-1]:][-1]+L
            not_work_set.append(s)
            not_work_set.append(e)
        else:
            s = not_work[start:points[i]][0]
            e = not_work[start:points[i]][-1]+L
            not_work_set.append(s)
            not_work_set.append(e)
            start = points[i]
    not_work_set.append(len(target)-1)  

    result = [[not_work_set[i],not_work_set[i+1]] for i in range(0,len(not_work_set),2)]
    result = filter(None, (map(lambda x: x if abs(x[1]-x[0]) > in_a_row else None, result)))
    return list(result)

def get_boundary(srs: pd.Series, n=2):
    n_sig = srs.std()*n
    lb = srs.mean() - n_sig 
    ub = srs.mean() + n_sig
    return lb, ub

