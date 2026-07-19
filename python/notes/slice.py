import cutlass
import cutlass.cute as cute


@cute.jit
def test():
    L = cute.make_layout((5, 8, 10, 13))
    sL = cute.slice_(L, coord=(1, None, 3, None))
    sL2 = cute.slice_(L, coord=(None, None, 3, None))
    print("L: {}\n".format(L))
    print("sL: {}\n".format(sL))
    print("sL2: {}\n".format(sL2))


if __name__ == "__main__":
    test()
