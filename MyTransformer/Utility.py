

def add_to_class(Class):
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


if __name__=='__main__':

    print('##### add_to_class() function test')

    class A:
        def __init__(self):
            self.b = 1
    a = A()

    @add_to_class(A)
    def do(self):
        print('Class attribute "b" is', self.b)

    a.do()







