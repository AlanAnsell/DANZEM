def FastApply(series, fn, as_list=False):
    fn_map = {}
    for x in series:
        if x not in fn_map:
            fn_map[x] = fn(x)
    
    if as_list:
        return [fn_map[x] for x in series]
    else:
        return series.apply(lambda x: fn_map[x])

