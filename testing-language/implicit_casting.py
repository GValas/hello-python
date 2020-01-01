class Toto:
    def __init__(self, name, age):
        self.name = name
        self.age = age


def display(T):
    print(T.name, T.age)


t = Toto('bob', 34)
display('bob', 34)
