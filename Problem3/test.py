def decode(a, b, start=0.0, finish=1.0):
    if start >= a or b >= finish:
        return []
    step = finish - start
    if start <= a and b <= start + step * 0.4:
        return ['a'] + decode(a, b, start, start + step * 0.4)
    elif start + step * 0.4 <= a and b <= start + step * 0.5:
        return ['b'] + decode(a, b, start + step * 0.4, start + step * 0.5)
    else:
        return ['c'] + decode(a, b, start + step * 0.5, finish)


print(decode(0.295, 0.3))
