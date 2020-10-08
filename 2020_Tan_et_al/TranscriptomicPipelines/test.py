class B:
    def __init__(self):
        self.c = 1

class A:
    def __init__(self, a, b = B()):
        self.a = 1
        self.b = b
        
        


        
if __name__ == "__main__":
    instance1 = A(1)
    instance2 = A(2)
    instance1.b.c = 2
    print(instance2.a)
    print(instance2.b.c)